from __future__ import annotations

import math
import time
from collections.abc import Iterable

import torch
from torch.nn import functional as F


@torch.inference_mode()
def perplexity(
    model: torch.nn.Module,
    token_ids: list[int],
    *,
    sequence_length: int = 512,
    device: torch.device | str = "cpu",
) -> float:
    """Sliding, non-overlapping causal-LM perplexity used for every precision point."""

    if len(token_ids) < 2:
        raise ValueError("perplexity requires at least two tokens")
    device = torch.device(device)
    negative_log_likelihood = 0.0
    predicted_tokens = 0
    for start in range(0, len(token_ids) - 1, sequence_length):
        chunk = token_ids[start : start + sequence_length + 1]
        if len(chunk) < 2:
            continue
        inputs = torch.tensor(chunk[:-1], device=device)[None]
        labels = torch.tensor(chunk[1:], device=device)
        logits = model(inputs)[0]
        loss = F.cross_entropy(logits.float(), labels, reduction="sum")
        negative_log_likelihood += float(loss.item())
        predicted_tokens += labels.numel()
    return math.exp(negative_log_likelihood / predicted_tokens)


@torch.inference_mode()
def measure_decode_throughput(
    model: torch.nn.Module,
    prompts: Iterable[list[int]],
    *,
    output_tokens: int = 32,
    device: torch.device | str = "cpu",
) -> float:
    """Matched greedy workload for relative precision comparisons."""

    device = torch.device(device)
    total = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for prompt in prompts:
        sequence = list(prompt)
        for _ in range(output_tokens):
            logits = model(torch.tensor(sequence, device=device)[None])[0, -1]
            sequence.append(int(logits.argmax().item()))
            total += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return total / max(time.perf_counter() - started, 1e-9)


def load_wikitext_tokens(tokenizer, *, split: str = "test", limit: int | None = None) -> list[int]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("install llm-serving-engine[eval] to load WikiText") from error
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row["text"] for row in rows if row["text"].strip())
    tokens = tokenizer.encode(text)
    return tokens[:limit] if limit else tokens


@torch.inference_mode()
def hellaswag_accuracy(
    model: torch.nn.Module,
    tokenizer,
    *,
    limit: int = 100,
    device: torch.device | str = "cpu",
) -> float:
    """Length-normalized completion likelihood on a reproducible HellaSwag subset."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("install llm-serving-engine[eval] to load HellaSwag") from error
    rows = load_dataset("Rowan/hellaswag", split=f"validation[:{limit}]")
    device = torch.device(device)
    correct = 0
    for row in rows:
        context_tokens = tokenizer.encode(row["ctx"])
        scores: list[float] = []
        for ending in row["endings"]:
            ending_tokens = tokenizer.encode(" " + ending)
            combined = context_tokens + ending_tokens
            inputs = torch.tensor(combined[:-1], device=device)[None]
            logits = model(inputs)[0]
            labels = torch.tensor(combined[1:], device=device)
            token_losses = F.cross_entropy(logits.float(), labels, reduction="none")
            start = max(len(context_tokens) - 1, 0)
            scores.append(-float(token_losses[start:].mean().item()))
        correct += int(max(range(len(scores)), key=scores.__getitem__) == int(row["label"]))
    return correct / max(len(rows), 1)
