from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass
from typing import Any

import torch

from llmserve.config import CacheConfig, SchedulerConfig
from llmserve.kv_cache import CacheFullError, PagedKVCache
from llmserve.model import Transformer
from llmserve.scheduler import IterationScheduler, RequestState, TokenCallback
from llmserve.tokenizer import Tokenizer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 7
    stop_token_ids: tuple[int, ...] = ()


def sample_token(logits: torch.Tensor, config: GenerationConfig, generator: torch.Generator) -> int:
    if config.temperature <= 0:
        return int(logits.argmax().item())
    probabilities = torch.softmax(logits.float() / config.temperature, dim=-1)
    if config.top_p < 1.0:
        sorted_probabilities, indices = torch.sort(probabilities, descending=True)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities >= config.top_p
        sorted_probabilities[remove] = 0
        sorted_probabilities /= sorted_probabilities.sum()
        selected = torch.multinomial(sorted_probabilities, 1, generator=generator)
        return int(indices[selected].item())
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


class NaiveDecoder:
    """Sequential reference implementation: recompute the full prefix for every token."""

    def __init__(self, model: Transformer, device: torch.device | str = "cpu") -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self._lock = asyncio.Lock()

    @torch.inference_mode()
    async def generate(
        self,
        prompt_token_ids: list[int],
        config: GenerationConfig,
        callback: TokenCallback | None = None,
    ) -> list[int]:
        async with self._lock:  # Phase 1 intentionally handles exactly one request at a time.
            output: list[int] = []
            generator = torch.Generator(device=self.device).manual_seed(config.seed)
            for _ in range(config.max_new_tokens):
                token_ids = torch.tensor([prompt_token_ids + output], device=self.device)
                logits = self.model(token_ids)[0, -1]
                token = sample_token(logits, config, generator)
                output.append(token)
                if callback:
                    result = callback(token)
                    if inspect.isawaitable(result):
                        await result
                if token in config.stop_token_ids:
                    break
                await asyncio.sleep(0)
            return output


class LLMEngine:
    """Async iteration-level engine backed by the custom paged attention implementation."""

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        *,
        cache_config: CacheConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
        scheduler_mode: str = "continuous",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        cache_config = cache_config or CacheConfig()
        scheduler_config = scheduler_config or SchedulerConfig()
        self.device = torch.device(device)
        self.model = model.to(device=self.device, dtype=dtype).eval()
        self.tokenizer = tokenizer
        self.cache = PagedKVCache(model.config, cache_config, device=self.device, dtype=dtype)
        self.scheduler = IterationScheduler(
            scheduler_config.max_batch_size, scheduler_config.policy, scheduler_mode
        )
        self.scheduler_config = scheduler_config
        self._incoming: asyncio.Queue[RequestState] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._generation_configs: dict[str, GenerationConfig] = {}
        self._generators: dict[str, torch.Generator] = {}

    async def generate(
        self,
        prompt: str | list[int],
        config: GenerationConfig | None = None,
        callback: TokenCallback | None = None,
        *,
        priority: int = 0,
        request_id: str | None = None,
    ) -> list[int]:
        config = config or GenerationConfig()
        if self._closed:
            raise RuntimeError("engine is closed")
        if config.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())
        prompt_tokens = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not prompt_tokens:
            raise ValueError("prompt cannot be empty")
        request_id = request_id or str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        state = RequestState(
            request_id=request_id,
            prompt_token_ids=prompt_tokens,
            max_new_tokens=config.max_new_tokens,
            callback=callback,
            priority=priority,
            future=future,
        )
        self._generation_configs[request_id] = config
        generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self._generators[request_id] = generator
        await self._incoming.put(state)
        return await future

    async def _run(self) -> None:
        try:
            while not self._closed:
                if self.scheduler.empty:
                    request = await self._incoming.get()
                    self.scheduler.add(request)
                while True:
                    try:
                        self.scheduler.add(self._incoming.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                for request in self.scheduler.admit():
                    self.cache.create(request.request_id)
                if self.scheduler.active:
                    await self._step()
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self._fail_all(error)

    @torch.inference_mode()
    async def _step(self) -> None:
        active = list(self.scheduler.active)
        required_blocks = sum(
            self.cache.sequences[item.request_id].length % self.cache.config.block_size == 0
            for item in active
        )
        while required_blocks > self.cache.allocator.free_blocks and len(active) > 1:
            victim = max(active, key=lambda item: item.processed_tokens)
            self.cache.free(victim.request_id)
            self.scheduler.preempt(victim)
            active.remove(victim)
            required_blocks = sum(
                self.cache.sequences[item.request_id].length % self.cache.config.block_size == 0
                for item in active
            )
        if required_blocks > self.cache.allocator.free_blocks:
            error = CacheFullError("one sequence exceeds total KV cache capacity")
            self._finish(active[0], error=error)
            return

        slots = [self.cache.reserve_token(item.request_id) for item in active]
        token_ids = torch.tensor([item.next_input_token for item in active], device=self.device)
        positions = torch.tensor([item.processed_tokens for item in active], device=self.device)
        logits = self.model.forward_paged(
            token_ids, positions, [item.request_id for item in active], slots, self.cache
        )
        completed: list[RequestState] = []
        for row, request in enumerate(active):
            request.processed_tokens += 1
            if request.processed_tokens < len(request.all_tokens):
                continue  # Replaying after preemption.
            config = self._generation_configs[request.request_id]
            token = sample_token(logits[row], config, self._generators[request.request_id])
            request.generated_token_ids.append(token)
            if request.callback:
                result = request.callback(token)
                if inspect.isawaitable(result):
                    await result
            if (
                len(request.generated_token_ids) >= request.max_new_tokens
                or token in config.stop_token_ids
            ):
                completed.append(request)
        for request in completed:
            self._finish(request)

    def _finish(self, request: RequestState, error: Exception | None = None) -> None:
        if request in self.scheduler.active:
            self.scheduler.finish(request)
        if request.request_id in self.cache.sequences:
            self.cache.free(request.request_id)
        self._generation_configs.pop(request.request_id, None)
        self._generators.pop(request.request_id, None)
        if request.future and not request.future.done():
            if error:
                request.future.set_exception(error)
            else:
                request.future.set_result(request.generated_token_ids)

    def _fail_all(self, error: Exception) -> None:
        for request in self.scheduler.active + self.scheduler.waiting:
            if request.future and not request.future.done():
                request.future.set_exception(error)

    async def close(self) -> None:
        self._closed = True
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    @property
    def cache_stats(self) -> dict[str, float]:
        return self.cache.stats(self.model.config.max_position_embeddings)

    async def __aenter__(self) -> LLMEngine:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
