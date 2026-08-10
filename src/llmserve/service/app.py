from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from llmserve.config import CacheConfig, ModelConfig, SchedulerConfig
from llmserve.engine import GenerationConfig, LLMEngine
from llmserve.model import Transformer
from llmserve.service.metrics import RequestObserver
from llmserve.tokenizer import ByteTokenizer


class CompletionRequest(BaseModel):
    model: str = "tiny-random"
    prompt: str
    max_tokens: int = Field(default=32, ge=1, le=2_048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    seed: int = 7


def _completion_payload(
    completion_id: str, model: str, text: str, prompt_tokens: int, output_tokens: int
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        },
    }


def create_app(engine: LLMEngine | None = None) -> FastAPI:
    owned_engine = engine is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is None:
            tokenizer = ByteTokenizer()
            app.state.engine = LLMEngine(
                Transformer(ModelConfig.tiny(tokenizer.vocab_size)),
                tokenizer,
                cache_config=CacheConfig(block_size=16, num_blocks=256),
                scheduler_config=SchedulerConfig(max_batch_size=16),
            )
        else:
            app.state.engine = engine
        yield
        if owned_engine:
            await app.state.engine.close()

    app = FastAPI(title="PyTorch LLM Serving Engine", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        serving_engine: LLMEngine = app.state.engine
        prompt_tokens = serving_engine.tokenizer.encode(request.prompt)
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        generation = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            stop_token_ids=(serving_engine.tokenizer.eos_token_id,),
        )
        observer = RequestObserver()
        if request.stream:
            return StreamingResponse(
                _stream_completion(
                    serving_engine,
                    request,
                    generation,
                    completion_id,
                    observer,
                ),
                media_type="text/event-stream",
            )
        try:
            output = await serving_engine.generate(
                prompt_tokens, generation, callback=lambda _: observer.token()
            )
            observer.finish("ok", False)
        except Exception as error:
            observer.finish("error", False)
            raise HTTPException(status_code=500, detail=str(error)) from error
        return JSONResponse(
            _completion_payload(
                completion_id,
                request.model,
                serving_engine.tokenizer.decode(output),
                len(prompt_tokens),
                len(output),
            )
        )

    return app


async def _stream_completion(
    engine: LLMEngine,
    request: CompletionRequest,
    generation: GenerationConfig,
    completion_id: str,
    observer: RequestObserver,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[int | Exception | None] = asyncio.Queue()

    async def on_token(token: int) -> None:
        observer.token()
        await queue.put(token)

    async def run() -> None:
        try:
            await engine.generate(request.prompt, generation, callback=on_token)
            await queue.put(None)
        except Exception as error:
            await queue.put(error)

    task = asyncio.create_task(run())
    try:
        while True:
            item = await queue.get()
            if item is None:
                observer.finish("ok", True)
                yield "data: [DONE]\n\n"
                break
            if isinstance(item, Exception):
                observer.finish("error", True)
                payload = {"error": {"message": str(item), "type": type(item).__name__}}
                yield f"data: {json.dumps(payload)}\n\n"
                break
            payload = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "text": engine.tokenizer.decode([item]),
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
    finally:
        if not task.done():
            task.cancel()


app = create_app()
