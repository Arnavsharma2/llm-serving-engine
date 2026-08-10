from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32_000
    hidden_size: int = 512
    intermediate_size: int = 1_376
    num_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 4_096
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def tiny(cls, vocab_size: int = 260) -> ModelConfig:
        return cls(
            vocab_size=vocab_size,
            hidden_size=128,
            intermediate_size=352,
            num_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1_024,
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        aliases = {
            "num_hidden_layers": "num_layers",
            "max_seq_len": "max_position_embeddings",
        }
        normalized = {aliases.get(key, key): value for key, value in values.items()}
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in normalized.items() if key in allowed})

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with Path(path).open() as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CacheConfig:
    block_size: int = 16
    num_blocks: int = 1_024


@dataclass(frozen=True)
class SchedulerConfig:
    max_batch_size: int = 16
    max_tokens_per_step: int = 2_048
    policy: str = "fcfs"
    preemption: str = "largest"
