#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/accuracy-throughput.png"))
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())["results"]
    baseline = rows[0]["perplexity"]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for row in rows:
        axis.scatter(row["tokens_per_second"], row["perplexity"], s=75)
        axis.annotate(
            row["precision"].upper(),
            (row["tokens_per_second"], row["perplexity"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axis.axhline(baseline, color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel("Decode throughput (tokens/s)")
    axis.set_ylabel("WikiText-2 perplexity (lower is better)")
    axis.set_title("Weight-only quantization tradeoff")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
