from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from llmserve.checkpoint import load_qwen2
from llmserve.evaluation import load_wikitext_tokens, perplexity
from llmserve.quantization import model_storage_bytes, quantize_model
from llmserve.tokenizer import HuggingFaceTokenizer


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release(candidate: torch.nn.Module | None) -> None:
    if candidate is not None:
        del candidate
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_candidate(model_id: str, backend: str, device: torch.device) -> torch.nn.Module:
    candidate = load_qwen2(model_id, device="cpu", dtype=torch.float16)
    if backend != "fp16":
        candidate = quantize_model(
            candidate,
            8,
            backend="reference" if backend == "reference-int8" else "triton",
            inplace=True,
            fallback_to_reference=False,
        )
    return candidate.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/checkpoint4-t4-accuracy.json")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint 4 accuracy evaluation requires CUDA")
    if args.stride != args.sequence_length:
        raise ValueError("this evaluator uses non-overlapping windows, so stride must equal length")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tokenizer = HuggingFaceTokenizer(args.model)
    token_ids = load_wikitext_tokens(
        tokenizer, split=args.dataset_split, limit=args.tokens
    )
    corpus_payload = json.dumps(token_ids, separators=(",", ":")).encode()
    rows: list[dict[str, object]] = []
    for backend in ("fp16", "reference-int8", "triton-int8"):
        release(None)
        started = time.perf_counter()
        candidate = load_candidate(args.model, backend, device)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        value = perplexity(
            candidate,
            token_ids,
            sequence_length=args.sequence_length,
            device=device,
        )
        rows.append(
            {
                "backend": backend,
                "perplexity": value,
                "model_storage_bytes": model_storage_bytes(candidate),
                "load_and_quantization_s": load_s,
            }
        )
        print("CHECKPOINT4_ACCURACY_ROW", json.dumps(rows[-1], sort_keys=True), flush=True)
        del candidate
        release(None)

    baseline = float(rows[0]["perplexity"])
    for row in rows:
        value = float(row["perplexity"])
        row["absolute_delta"] = value - baseline
        row["percentage_delta"] = (value / baseline - 1.0) * 100
    artifact = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "source_revision": args.source_revision
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
        "corpus": {
            "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
            "split": args.dataset_split,
            "model": args.model,
            "model_revision": args.model_revision,
            "tokenizer": args.model,
            "tokenizer_revision": args.model_revision,
            "tokens": len(token_ids),
            "sequence_length": args.sequence_length,
            "stride": args.stride,
            "seed": args.seed,
            "token_ids_sha256": hashlib.sha256(corpus_payload).hexdigest(),
        },
        "command": " ".join([sys.executable, *sys.argv]),
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    checksum_path = output.with_suffix(".sha256")
    checksum_path.write_text(f"{checksum(output)}  {output.name}\n")
    print("CHECKPOINT4_ACCURACY_CHECKSUM", checksum_path.read_text(), flush=True)


if __name__ == "__main__":
    main()
