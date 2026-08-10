from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    eos_token_id: int
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


class ByteTokenizer:
    """Dependency-free tokenizer for smoke tests and the bundled random demo model."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 260

    def encode(self, text: str) -> list[int]:
        return [byte + 4 for byte in text.encode("utf-8")]

    def decode(self, token_ids: list[int]) -> str:
        payload = bytes(token - 4 for token in token_ids if 4 <= token < 260)
        return payload.decode("utf-8", errors="replace")


class HuggingFaceTokenizer:
    def __init__(self, model_id_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError("install llm-serving-engine[models] for HF tokenizers") from error
        self.inner = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=False)
        self.eos_token_id = int(self.inner.eos_token_id)
        self.vocab_size = int(self.inner.vocab_size)

    def encode(self, text: str) -> list[int]:
        return list(self.inner.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return self.inner.decode(token_ids, skip_special_tokens=True)
