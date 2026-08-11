from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path


def _device(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _load_runtime(args):
    import torch

    from llmserve.checkpoint import load_qwen2
    from llmserve.config import ModelConfig
    from llmserve.model import Transformer
    from llmserve.tokenizer import ByteTokenizer, HuggingFaceTokenizer

    device = _device(args.device)
    if args.model:
        dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
        model = load_qwen2(args.model, device=str(device), dtype=dtype)
        tokenizer = HuggingFaceTokenizer(args.model)
    else:
        tokenizer = ByteTokenizer()
        torch.manual_seed(args.seed)
        model = Transformer(ModelConfig.tiny(tokenizer.vocab_size)).to(device).eval()
    return model, tokenizer, device


async def _benchmark(args) -> None:
    from llmserve.benchmark import BenchmarkHarness, LoadProfile
    from llmserve.config import CacheConfig, SchedulerConfig
    from llmserve.engine import GenerationConfig, LLMEngine, NaiveDecoder

    model, tokenizer, device = _load_runtime(args)
    profile = LoadProfile.from_yaml(args.profile)
    profile = replace(
        profile,
        requests=args.requests or profile.requests,
        arrival_rate=args.arrival_rate or profile.arrival_rate,
        vocabulary_size=model.config.vocab_size,
        prompt=replace(
            profile.prompt,
            maximum=min(
                profile.prompt.maximum,
                model.config.max_position_embeddings - profile.output.maximum,
            ),
        ),
    )
    harness = BenchmarkHarness(profile)

    def generation_for(spec):
        return GenerationConfig(
            max_new_tokens=spec.max_new_tokens,
            temperature=0.0,
            seed=args.seed,
            stop_token_ids=(),
        )

    engine = None
    if args.config == "naive":
        decoder = NaiveDecoder(model, device)

        async def generate(spec, callback):
            return await decoder.generate(
                list(spec.prompt_token_ids), generation_for(spec), callback
            )

    else:
        mode = "static" if args.config == "static" else "continuous"
        batch_size = 1 if args.config == "paged" else args.batch_size
        engine = LLMEngine(
            model,
            tokenizer,
            cache_config=CacheConfig(block_size=args.block_size, num_blocks=args.num_blocks),
            scheduler_config=SchedulerConfig(
                max_batch_size=batch_size,
                max_tokens_per_step=args.max_tokens_per_step,
                policy=args.policy,
            ),
            scheduler_mode=mode,
            paged_attention_backend=args.paged_attention_backend,
            prefill_chunk_size=args.prefill_chunk_size,
            cache_device_metadata=not args.rebuild_device_metadata,
            collect_iteration_metrics=True,
            device=device,
            dtype=next(model.parameters()).dtype,
        )

        async def generate(spec, callback):
            return await engine.generate(
                list(spec.prompt_token_ids),
                generation_for(spec),
                callback,
                request_id=spec.request_id,
            )

    report = await harness.run(f"{args.config}-{args.policy}", generate)
    if engine:
        report.cache = engine.cache_stats
        report.metadata["engine"] = engine.iteration_stats
        await engine.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(output)
    report.write_csv(output.with_suffix(".csv"))
    print(json.dumps(report.summary, indent=2))


def _quantize_eval(args) -> None:
    from llmserve.evaluation import (
        hellaswag_accuracy,
        load_wikitext_tokens,
        measure_decode_throughput,
        perplexity,
    )
    from llmserve.quantization import model_storage_bytes, quantize_model

    model, tokenizer, device = _load_runtime(args)
    if args.text_file:
        tokens = tokenizer.encode(Path(args.text_file).read_text())[: args.eval_tokens]
    else:
        tokens = load_wikitext_tokens(tokenizer, limit=args.eval_tokens)
    prompts = [
        tokens[index : index + args.prompt_length]
        for index in range(0, 4 * args.prompt_length, args.prompt_length)
    ]
    rows = []
    for label, candidate in (
        ("fp", model),
        ("int8", quantize_model(model, 8)),
        ("int4", quantize_model(model, 4)),
    ):
        candidate.to(device).eval()
        row = {
            "precision": label,
            "perplexity": perplexity(
                candidate, tokens, sequence_length=args.sequence_length, device=device
            ),
            "hellaswag_accuracy": hellaswag_accuracy(
                candidate, tokenizer, limit=args.hellaswag_samples, device=device
            )
            if args.hellaswag_samples
            else None,
            "tokens_per_second": measure_decode_throughput(
                candidate, prompts, output_tokens=args.output_tokens, device=device
            ),
            "model_size_gb": model_storage_bytes(candidate) / 1024**3,
        }
        rows.append(row)
        print(json.dumps(row))
    baseline = rows[0]
    for row in rows:
        row["perplexity_delta"] = row["perplexity"] - baseline["perplexity"]
        if row["hellaswag_accuracy"] is not None:
            row["accuracy_delta"] = row["hellaswag_accuracy"] - baseline["hellaswag_accuracy"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": rows}, indent=2) + "\n")


def _speculative_eval(args) -> None:
    import torch

    from llmserve.config import ModelConfig
    from llmserve.model import Transformer
    from llmserve.speculative import SpeculativeDecoder
    from llmserve.tokenizer import ByteTokenizer

    tokenizer = ByteTokenizer()
    torch.manual_seed(args.seed)
    target = Transformer(ModelConfig.tiny(tokenizer.vocab_size))
    # The smaller draft is independently initialized unless a trained draft is supplied later.
    draft_config = replace(
        ModelConfig.tiny(tokenizer.vocab_size),
        hidden_size=64,
        intermediate_size=176,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    draft = Transformer(draft_config)
    decoder = SpeculativeDecoder(draft, target, speculation_tokens=args.speculation_tokens)
    result = decoder.generate(tokenizer.encode(args.prompt), args.max_tokens)
    print(
        json.dumps(
            {
                "text": tokenizer.decode(result.token_ids),
                "acceptance_rate": result.acceptance_rate,
                "proposed_tokens": result.proposed_tokens,
                "accepted_tokens": result.accepted_tokens,
                "target_passes": result.target_passes,
            },
            indent=2,
        )
    )


def _serve(args) -> None:
    import torch
    import uvicorn

    from llmserve.config import CacheConfig, SchedulerConfig
    from llmserve.engine import LLMEngine
    from llmserve.service import create_app

    model, tokenizer, device = _load_runtime(args)
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    engine = LLMEngine(
        model,
        tokenizer,
        cache_config=CacheConfig(block_size=args.block_size, num_blocks=args.num_blocks),
        scheduler_config=SchedulerConfig(
            max_batch_size=args.batch_size,
            max_tokens_per_step=args.max_tokens_per_step,
            policy=args.policy,
        ),
        paged_attention_backend=args.paged_attention_backend,
        prefill_chunk_size=args.prefill_chunk_size,
        device=device,
        dtype=dtype,
    )
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level=args.log_level)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmserve")
    commands = parser.add_subparsers(dest="command", required=True)

    benchmark = commands.add_parser("benchmark", help="run the Phase 0 harness")
    benchmark.add_argument(
        "--config", choices=("naive", "paged", "static", "continuous"), default="continuous"
    )
    benchmark.add_argument("--policy", choices=("fcfs", "sjf", "priority"), default="fcfs")
    benchmark.add_argument("--profile", default="configs/benchmark-sharegpt.yaml")
    benchmark.add_argument("--model", help="local path or Hugging Face Qwen2/Qwen2.5 id")
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--requests", type=int)
    benchmark.add_argument("--arrival-rate", type=float)
    benchmark.add_argument("--batch-size", type=int, default=16)
    benchmark.add_argument("--block-size", type=int, default=16)
    benchmark.add_argument("--num-blocks", type=int, default=1024)
    benchmark.add_argument("--prefill-chunk-size", type=int, default=16)
    benchmark.add_argument("--max-tokens-per-step", type=int, default=2048)
    benchmark.add_argument("--rebuild-device-metadata", action="store_true")
    benchmark.add_argument(
        "--paged-attention-backend", choices=("pytorch", "triton"), default="pytorch"
    )
    benchmark.add_argument("--seed", type=int, default=7)
    benchmark.add_argument("--output", default="artifacts/benchmark.json")

    quantize = commands.add_parser(
        "quantize-eval", help="build FP/INT8/INT4 accuracy-throughput curve"
    )
    quantize.add_argument("--model", help="local path or Hugging Face Qwen2/Qwen2.5 id")
    quantize.add_argument("--device", default="auto")
    quantize.add_argument("--seed", type=int, default=7)
    quantize.add_argument("--text-file")
    quantize.add_argument("--eval-tokens", type=int, default=8192)
    quantize.add_argument("--sequence-length", type=int, default=512)
    quantize.add_argument("--prompt-length", type=int, default=64)
    quantize.add_argument("--output-tokens", type=int, default=32)
    quantize.add_argument("--hellaswag-samples", type=int, default=100)
    quantize.add_argument("--output", default="artifacts/quantization.json")

    speculative = commands.add_parser("speculative-eval", help="report greedy draft acceptance")
    speculative.add_argument("--prompt", default="Explain paged attention in one paragraph.")
    speculative.add_argument("--max-tokens", type=int, default=64)
    speculative.add_argument("--speculation-tokens", type=int, default=4)
    speculative.add_argument("--seed", type=int, default=7)

    serve = commands.add_parser("serve", help="start the OpenAI-compatible HTTP service")
    serve.add_argument("--model", help="local path or Hugging Face Qwen2/Qwen2.5 id")
    serve.add_argument("--device", default="auto")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--batch-size", type=int, default=16)
    serve.add_argument("--block-size", type=int, default=16)
    serve.add_argument("--num-blocks", type=int, default=1024)
    serve.add_argument("--prefill-chunk-size", type=int, default=16)
    serve.add_argument("--max-tokens-per-step", type=int, default=2048)
    serve.add_argument(
        "--paged-attention-backend", choices=("pytorch", "triton"), default="pytorch"
    )
    serve.add_argument("--policy", choices=("fcfs", "sjf", "priority"), default="fcfs")
    serve.add_argument("--log-level", default="info")
    serve.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "benchmark":
        asyncio.run(_benchmark(args))
    elif args.command == "quantize-eval":
        _quantize_eval(args)
    elif args.command == "speculative-eval":
        _speculative_eval(args)
    else:
        _serve(args)


if __name__ == "__main__":
    main()
