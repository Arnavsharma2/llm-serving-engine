from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from llmserve.config import CacheConfig, ModelConfig


class CacheFullError(RuntimeError):
    pass


@dataclass
class SequenceCache:
    sequence_id: str
    block_table: list[int] = field(default_factory=list)
    length: int = 0
    last_access_tick: int = 0


@dataclass(frozen=True)
class CacheSlot:
    block_id: int
    offset: int
    position: int


class BlockAllocator:
    """O(1) fixed-block allocator with explicit ownership accounting."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free = deque(range(num_blocks))
        self._owner: dict[int, str] = {}

    def allocate(self, owner: str) -> int:
        if not self._free:
            raise CacheFullError("paged KV cache has no free blocks")
        block = self._free.popleft()
        self._owner[block] = owner
        return block

    def free(self, block: int, owner: str) -> None:
        actual_owner = self._owner.get(block)
        if actual_owner != owner:
            raise ValueError(f"block {block} belongs to {actual_owner!r}, not {owner!r}")
        del self._owner[block]
        self._free.append(block)

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return len(self._owner)


class PagedKVCache:
    """Layer-major KV tensors plus per-sequence logical-to-physical block tables."""

    def __init__(
        self,
        model: ModelConfig,
        cache: CacheConfig,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model_config = model
        self.config = cache
        self.device = torch.device(device)
        self.dtype = dtype
        self.allocator = BlockAllocator(cache.num_blocks)
        self.sequences: dict[str, SequenceCache] = {}
        self._tick = 0
        self._high_water = {"blocks": 0, "used_tokens": 0, "sequences": 0}
        shape = (
            model.num_layers,
            2,
            cache.num_blocks,
            cache.block_size,
            model.num_key_value_heads,
            model.head_dim,
        )
        self.storage = torch.empty(shape, device=self.device, dtype=dtype)

    def create(self, sequence_id: str) -> None:
        if sequence_id in self.sequences:
            raise ValueError(f"sequence {sequence_id!r} already exists")
        self.sequences[sequence_id] = SequenceCache(sequence_id)

    def reserve_token(self, sequence_id: str) -> CacheSlot:
        sequence = self.sequences[sequence_id]
        position = sequence.length
        offset = position % self.config.block_size
        if offset == 0:
            sequence.block_table.append(self.allocator.allocate(sequence_id))
        slot = CacheSlot(sequence.block_table[position // self.config.block_size], offset, position)
        sequence.length += 1
        self._touch(sequence)
        if self.allocator.used_blocks >= self._high_water["blocks"]:
            self._high_water = {
                "blocks": self.allocator.used_blocks,
                "used_tokens": sum(item.length for item in self.sequences.values()),
                "sequences": len(self.sequences),
            }
        return slot

    def write(
        self,
        layer: int,
        slots: list[CacheSlot],
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write one token per sequence; keys/values have shape [batch, kv_heads, head_dim]."""

        block_ids = torch.tensor([slot.block_id for slot in slots], device=self.device)
        offsets = torch.tensor([slot.offset for slot in slots], device=self.device)
        self.storage[layer, 0, block_ids, offsets] = keys
        self.storage[layer, 1, block_ids, offsets] = values

    def gather(
        self, layer: int, sequence_ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize active logical blocks into a padded batch for attention."""

        if not sequence_ids:
            raise ValueError("sequence_ids cannot be empty")
        lengths = torch.tensor(
            [self.sequences[sequence_id].length for sequence_id in sequence_ids],
            device=self.device,
            dtype=torch.long,
        )
        max_length = int(lengths.max().item())
        keys = torch.zeros(
            len(sequence_ids),
            max_length,
            self.model_config.num_key_value_heads,
            self.model_config.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        values = torch.zeros_like(keys)
        for row, sequence_id in enumerate(sequence_ids):
            sequence = self.sequences[sequence_id]
            remaining = sequence.length
            cursor = 0
            for block in sequence.block_table:
                take = min(remaining, self.config.block_size)
                keys[row, cursor : cursor + take] = self.storage[layer, 0, block, :take]
                values[row, cursor : cursor + take] = self.storage[layer, 1, block, :take]
                cursor += take
                remaining -= take
                if remaining == 0:
                    break
            self._touch(sequence)
        return keys, values, lengths

    def iter_kv_blocks(self, layer: int, sequence_id: str):
        """Yield valid K/V views in logical order without making a contiguous cache copy."""

        sequence = self.sequences[sequence_id]
        remaining = sequence.length
        for block in sequence.block_table:
            take = min(remaining, self.config.block_size)
            yield self.storage[layer, 0, block, :take], self.storage[layer, 1, block, :take]
            remaining -= take
            if remaining == 0:
                break
        self._touch(sequence)

    def free(self, sequence_id: str) -> None:
        sequence = self.sequences.pop(sequence_id)
        for block in sequence.block_table:
            self.allocator.free(block, sequence_id)

    def evict(self, policy: str = "largest", exclude: set[str] | None = None) -> tuple[str, int]:
        candidates = [
            sequence
            for sequence in self.sequences.values()
            if not exclude or sequence.sequence_id not in exclude
        ]
        if not candidates:
            raise CacheFullError("no evictable sequences")
        if policy == "largest":
            victim = max(candidates, key=lambda item: (item.length, -item.last_access_tick))
        elif policy == "lru":
            victim = min(candidates, key=lambda item: item.last_access_tick)
        else:
            raise ValueError(f"unknown eviction policy: {policy}")
        recompute = victim.length
        victim_id = victim.sequence_id
        self.free(victim_id)
        return victim_id, recompute

    def _touch(self, sequence: SequenceCache) -> None:
        self._tick += 1
        sequence.last_access_tick = self._tick

    @property
    def bytes_per_block(self) -> int:
        return (
            self.model_config.num_layers
            * 2
            * self.config.block_size
            * self.model_config.num_key_value_heads
            * self.model_config.head_dim
            * self.storage.element_size()
        )

    def stats(self, contiguous_max_length: int | None = None) -> dict[str, float]:
        used_tokens = sum(sequence.length for sequence in self.sequences.values())
        allocated_slots = self.allocator.used_blocks * self.config.block_size
        paged_waste_slots = allocated_slots - used_tokens
        result = {
            "blocks_used": float(self.allocator.used_blocks),
            "blocks_free": float(self.allocator.free_blocks),
            "used_tokens": float(used_tokens),
            "paged_waste_tokens": float(paged_waste_slots),
            "paged_fragmentation_pct": 100.0 * paged_waste_slots / max(allocated_slots, 1),
            "paged_allocated_gb": self.allocator.used_blocks * self.bytes_per_block / 1024**3,
            "peak_blocks_used": float(self._high_water["blocks"]),
            "peak_paged_allocated_gb": self._high_water["blocks"] * self.bytes_per_block / 1024**3,
        }
        if contiguous_max_length is not None:
            high_water_tokens = self._high_water["used_tokens"]
            contiguous_slots = self._high_water["sequences"] * contiguous_max_length
            contiguous_waste = max(contiguous_slots - high_water_tokens, 0)
            paged_slots_at_peak = self._high_water["blocks"] * self.config.block_size
            bytes_per_token = self.bytes_per_block / self.config.block_size
            result.update(
                contiguous_waste_tokens=float(contiguous_waste),
                contiguous_waste_gb=contiguous_waste * bytes_per_token / 1024**3,
                memory_reduction_pct=100.0
                * (contiguous_slots - paged_slots_at_peak)
                / max(contiguous_slots, 1),
            )
        return result

    @property
    def capacity_tokens(self) -> int:
        return self.config.num_blocks * self.config.block_size
