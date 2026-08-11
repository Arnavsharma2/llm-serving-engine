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
from llmserve.triton_kernels import is_triton_paged_attention_available


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


def sample_tokens(
    logits: torch.Tensor,
    configs: list[GenerationConfig],
    generators: list[torch.Generator],
) -> list[int]:
    """Select a batch of tokens with one device-to-host synchronization."""

    if logits.ndim != 2 or logits.shape[0] != len(configs) or len(configs) != len(generators):
        raise ValueError("batched sampling inputs must have matching rows")
    if all(config.temperature <= 0 for config in configs):
        return logits.argmax(dim=-1).to(device="cpu").tolist()

    selected: list[torch.Tensor] = []
    for row, (config, generator) in enumerate(zip(configs, generators, strict=True)):
        if config.temperature <= 0:
            selected.append(logits[row].argmax())
            continue
        probabilities = torch.softmax(logits[row].float() / config.temperature, dim=-1)
        if config.top_p < 1.0:
            sorted_probabilities, indices = torch.sort(probabilities, descending=True)
            cumulative = sorted_probabilities.cumsum(dim=-1)
            remove = cumulative - sorted_probabilities >= config.top_p
            sorted_probabilities[remove] = 0
            sorted_probabilities /= sorted_probabilities.sum()
            sampled = torch.multinomial(sorted_probabilities, 1, generator=generator)
            selected.append(indices[sampled][0])
        else:
            selected.append(torch.multinomial(probabilities, 1, generator=generator)[0])
    return torch.stack(selected).to(device="cpu").tolist()


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
        paged_attention_backend: str = "pytorch",
        prefill_chunk_size: int = 16,
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
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if scheduler_config.max_tokens_per_step < scheduler_config.max_batch_size:
            raise ValueError("max_tokens_per_step must allow one token per active request")
        self.prefill_chunk_size = prefill_chunk_size
        if paged_attention_backend not in {"pytorch", "triton"}:
            raise ValueError("paged_attention_backend must be pytorch or triton")
        if paged_attention_backend == "triton":
            if dtype != torch.float16:
                raise TypeError("Triton paged attention requires dtype=torch.float16")
            if cache_config.block_size != 16:
                raise ValueError("Triton paged attention requires cache block_size=16")
            if not is_triton_paged_attention_available(self.device):
                raise RuntimeError(
                    "Triton paged attention requires a CUDA GPU with compute capability 7.5+"
                )
        self.paged_attention_backend = paged_attention_backend
        self._incoming: asyncio.Queue[RequestState] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._generation_configs: dict[str, GenerationConfig] = {}
        self._generators: dict[str, torch.Generator] = {}
        iteration_capacity = scheduler_config.max_tokens_per_step
        pinned = self.device.type == "cuda"
        self._host_token_ids = torch.empty(
            iteration_capacity, device="cpu", dtype=torch.long, pin_memory=pinned
        )
        self._host_positions = torch.empty(
            iteration_capacity, device="cpu", dtype=torch.long, pin_memory=pinned
        )
        self._token_ids = torch.empty(iteration_capacity, device=self.device, dtype=torch.long)
        self._positions = torch.empty(iteration_capacity, device=self.device, dtype=torch.long)
        self._iteration_metadata = self.cache.allocate_iteration_metadata(iteration_capacity)

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
        planned = self._plan_iteration(list(self.scheduler.active))
        required_blocks = self._required_blocks(planned)
        while required_blocks > self.cache.allocator.free_blocks and len(planned) > 1:
            victim, _ = max(planned, key=lambda item: item[0].processed_tokens)
            self.cache.free(victim.request_id)
            self.scheduler.preempt(victim)
            planned = [item for item in planned if item[0] is not victim]
            required_blocks = self._required_blocks(planned)
        if required_blocks > self.cache.allocator.free_blocks:
            error = CacheFullError("one sequence exceeds total KV cache capacity")
            self._finish(planned[0][0], error=error)
            return

        sequence_ids: list[str] = []
        context_lengths: list[int] = []
        slots = []
        final_rows: list[int] = []
        cursor = 0
        for request, count in planned:
            request_tokens = request.all_tokens[
                request.processed_tokens : request.processed_tokens + count
            ]
            request_slots = self.cache.reserve_tokens(request.request_id, count)
            for token, slot in zip(request_tokens, request_slots, strict=True):
                self._host_token_ids[cursor] = token
                self._host_positions[cursor] = slot.position
                sequence_ids.append(request.request_id)
                context_lengths.append(slot.position + 1)
                slots.append(slot)
                cursor += 1
            final_rows.append(cursor - 1)

        self._token_ids[:cursor].copy_(
            self._host_token_ids[:cursor], non_blocking=self.device.type == "cuda"
        )
        self._positions[:cursor].copy_(
            self._host_positions[:cursor], non_blocking=self.device.type == "cuda"
        )
        block_table_rows, sequence_lengths, block_ids, offsets = (
            self.cache.prepare_iteration_metadata(
                self._iteration_metadata, sequence_ids, slots, context_lengths
            )
        )
        logits = self.model.forward_paged(
            self._token_ids[:cursor],
            self._positions[:cursor],
            sequence_ids,
            slots,
            self.cache,
            backend=self.paged_attention_backend,
            block_table_rows=block_table_rows,
            sequence_lengths=sequence_lengths,
            context_lengths=context_lengths,
            block_ids=block_ids,
            offsets=offsets,
        )
        sample_requests: list[RequestState] = []
        sample_rows: list[int] = []
        for (request, count), row in zip(planned, final_rows, strict=True):
            request.processed_tokens += count
            if request.processed_tokens < len(request.all_tokens):
                continue  # Replaying after preemption.
            sample_requests.append(request)
            sample_rows.append(row)

        if not sample_requests:
            return
        selected_logits = torch.stack([logits[row] for row in sample_rows])
        configs = [self._generation_configs[request.request_id] for request in sample_requests]
        generators = [self._generators[request.request_id] for request in sample_requests]
        tokens = sample_tokens(selected_logits, configs, generators)
        completed: list[RequestState] = []
        for request, config, token in zip(sample_requests, configs, tokens, strict=True):
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

    def _plan_iteration(self, active: list[RequestState]) -> list[tuple[RequestState, int]]:
        """Give every active request progress, then spend remaining budget on prefill chunks."""

        budget = self.scheduler_config.max_tokens_per_step
        planned: list[tuple[RequestState, int]] = []
        for request in active:
            remaining = len(request.all_tokens) - request.processed_tokens
            if remaining <= 0:
                raise RuntimeError("active request has no input token to process")
            planned.append((request, 1))
            budget -= 1
        for index, (request, count) in enumerate(planned):
            if budget <= 0:
                break
            remaining = len(request.all_tokens) - request.processed_tokens
            desired = min(self.prefill_chunk_size, remaining)
            extra = min(desired - count, budget)
            planned[index] = (request, count + extra)
            budget -= extra
        return planned

    def _required_blocks(self, planned: list[tuple[RequestState, int]]) -> int:
        block_size = self.cache.config.block_size
        required = 0
        for request, count in planned:
            length = self.cache.sequences[request.request_id].length
            required += (length + count + block_size - 1) // block_size
            required -= (length + block_size - 1) // block_size
        return required

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
        for sequence_id in list(self.cache.sequences):
            self.cache.free(sequence_id)
        self.scheduler.active.clear()
        self.scheduler.waiting.clear()
        self._generation_configs.clear()
        self._generators.clear()

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
