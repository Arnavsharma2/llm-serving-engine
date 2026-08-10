from __future__ import annotations

import json
from pathlib import Path

import torch

from llmserve.config import ModelConfig
from llmserve.model import Transformer


def _resolve_model_path(model_id_or_path: str) -> Path:
    path = Path(model_id_or_path)
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError("install llm-serving-engine[models] to download checkpoints") from error
    return Path(
        snapshot_download(
            model_id_or_path,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab*", "merges*"],
        )
    )


def load_qwen2(
    model_id_or_path: str, *, device: str = "cpu", dtype: torch.dtype | None = None
) -> Transformer:
    """Load Qwen2/Qwen2.5 safetensors directly into this project's decoder implementation."""

    path = _resolve_model_path(model_id_or_path)
    config_values = json.loads((path / "config.json").read_text())
    if config_values.get("model_type") != "qwen2":
        raise ValueError("only Qwen2/Qwen2.5 checkpoints are supported by the direct loader")
    config = ModelConfig.from_dict(config_values)
    model = Transformer(config)
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError("install llm-serving-engine[models] for safetensors") from error

    destination = model.state_dict()
    loaded: set[str] = set()
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files under {path}")
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as source:
            for source_name in source.keys():
                target_name = source_name.removeprefix("model.")
                if target_name in destination:
                    tensor = source.get_tensor(source_name)
                    if destination[target_name].shape != tensor.shape:
                        raise ValueError(
                            f"shape mismatch for {source_name}: "
                            f"{tuple(tensor.shape)} != {tuple(destination[target_name].shape)}"
                        )
                    destination[target_name].copy_(tensor)
                    loaded.add(target_name)
    missing = set(destination) - loaded
    # A tied lm_head may be absent because it aliases embed_tokens.
    if config.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} tensors: {sorted(missing)[:5]}")
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device).eval()
