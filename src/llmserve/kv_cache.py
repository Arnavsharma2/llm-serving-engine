from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from llmserve.config import CacheConfig, ModelConfig


class CacheFullError(RuntimeError):
    pass


@dataclass
class SequenceCache:
    sequence_id: str
    metadata_row: int
    block_table: list[int] = field(default_factory=list)
    length: int = 0
    last_access_tick: int = 0


@dataclass(frozen=True)
class CacheSlot:
    block_id: int
    offset: int
    position: int


@dataclass
class IterationMetadata:
    """Reusable host/device buffers shared by every decoder layer in an iteration."""

    capacity: int
    host_block_table_rows: torch.Tensor
    host_context_lengths: torch.Tensor
    host_block_ids: torch.Tensor
    host_offsets: torch.Tensor
    block_table_rows: torch.Tensor
    context_lengths: torch.Tensor
    block_ids: torch.Tensor
    offsets: torch.Tensor

    def active(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if count < 0 or count > self.capacity:
            raise ValueError("active metadata count exceeds buffer capacity")
        return (
            self.block_table_rows[:count],
            self.context_lengths[:count],
            self.block_ids[:count],
            self.offsets[:count],
        )


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
        self.max_blocks_per_sequence = min(
            cache.num_blocks,
            math.ceil(model.max_position_embeddings / cache.block_size),
        )
        self._free_metadata_rows = deque(range(cache.num_blocks))
        self._tick = 0
        self.metadata_rebuild_ms = 0.0
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
        self.device_block_tables = torch.full(
            (cache.num_blocks, self.max_blocks_per_sequence),
            -1,
            device=self.device,
            dtype=torch.int32,
        )
        self.device_context_lengths = torch.zeros(
            cache.num_blocks, device=self.device, dtype=torch.int32
        )

    def create(self, sequence_id: str) -> None:
        if sequence_id in self.sequences:
            raise ValueError(f"sequence {sequence_id!r} already exists")
        if not self._free_metadata_rows:
            raise CacheFullError("paged KV cache has no free metadata rows")
        metadata_row = self._free_metadata_rows.popleft()
        self.device_block_tables[metadata_row].fill_(-1)
        self.device_context_lengths[metadata_row] = 0
        self.sequences[sequence_id] = SequenceCache(sequence_id, metadata_row)

    def reserve_token(self, sequence_id: str) -> CacheSlot:
        slot = self._reserve_token(sequence_id)
        sequence = self.sequences[sequence_id]
        self.device_context_lengths[sequence.metadata_row] = sequence.length
        return slot

    def _reserve_token(self, sequence_id: str) -> CacheSlot:
        sequence = self.sequences[sequence_id]
        position = sequence.length
        offset = position % self.config.block_size
        if offset == 0:
            logical_block = len(sequence.block_table)
            if logical_block >= self.max_blocks_per_sequence:
                raise CacheFullError("sequence exceeds paged KV metadata capacity")
            physical_block = self.allocator.allocate(sequence_id)
            sequence.block_table.append(physical_block)
            self.device_block_tables[sequence.metadata_row, logical_block] = physical_block
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

    def reserve_tokens(self, sequence_id: str, count: int) -> list[CacheSlot]:
        if count <= 0:
            raise ValueError("count must be positive")
        slots = [self._reserve_token(sequence_id) for _ in range(count)]
        sequence = self.sequences[sequence_id]
        self.device_context_lengths[sequence.metadata_row] = sequence.length
        return slots

    def allocate_iteration_metadata(self, capacity: int) -> IterationMetadata:
        if capacity <= 0:
            raise ValueError("iteration metadata capacity must be positive")

        def host_buffer() -> torch.Tensor:
            return torch.empty(
                capacity,
                device="cpu",
                dtype=torch.int32,
                pin_memory=self.device.type == "cuda",
            )

        def device_buffer() -> torch.Tensor:
            return torch.empty(capacity, device=self.device, dtype=torch.int32)

        return IterationMetadata(
            capacity=capacity,
            host_block_table_rows=host_buffer(),
            host_context_lengths=host_buffer(),
            host_block_ids=host_buffer(),
            host_offsets=host_buffer(),
            block_table_rows=device_buffer(),
            context_lengths=device_buffer(),
            block_ids=device_buffer(),
            offsets=device_buffer(),
        )

    def prepare_iteration_metadata(
        self,
        metadata: IterationMetadata,
        sequence_ids: list[str],
        slots: list[CacheSlot],
        context_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = len(sequence_ids)
        if count == 0:
            raise ValueError("sequence_ids cannot be empty")
        if len(slots) != count or len(context_lengths) != count:
            raise ValueError("iteration metadata inputs must have matching lengths")
        if count > metadata.capacity:
            raise ValueError("iteration metadata exceeds reusable buffer capacity")
        for index, (sequence_id, slot, context_length) in enumerate(
            zip(sequence_ids, slots, context_lengths, strict=True)
        ):
            sequence = self.sequences[sequence_id]
            if context_length <= 0 or context_length > sequence.length:
                raise ValueError("context length must reference reserved sequence tokens")
            metadata.host_block_table_rows[index] = sequence.metadata_row
            metadata.host_context_lengths[index] = context_length
            metadata.host_block_ids[index] = slot.block_id
            metadata.host_offsets[index] = slot.offset

        host_buffers = (
            metadata.host_block_table_rows,
            metadata.host_context_lengths,
            metadata.host_block_ids,
            metadata.host_offsets,
        )
        device_buffers = (
            metadata.block_table_rows,
            metadata.context_lengths,
            metadata.block_ids,
            metadata.offsets,
        )
        for host, device in zip(host_buffers, device_buffers, strict=True):
            device[:count].copy_(host[:count], non_blocking=self.device.type == "cuda")
        return metadata.active(count)

    def write(
        self,
        layer: int,
        slots: list[CacheSlot],
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        block_ids: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> None:
        """Write one token per sequence; keys/values have shape [batch, kv_heads, head_dim]."""

        if block_ids is None:
            block_ids = torch.tensor([slot.block_id for slot in slots], device=self.device)
        if offsets is None:
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

    def iter_kv_blocks(self, layer: int, sequence_id: str, length: int | None = None):
        """Yield valid K/V views in logical order without making a contiguous cache copy."""

        sequence = self.sequences[sequence_id]
        remaining = sequence.length if length is None else length
        if remaining <= 0 or remaining > sequence.length:
            raise ValueError("requested KV length is outside the cached sequence")
        for block in sequence.block_table:
            take = min(remaining, self.config.block_size)
            yield self.storage[layer, 0, block, :take], self.storage[layer, 1, block, :take]
            remaining -= take
            if remaining == 0:
                break
        self._touch(sequence)

    def block_table_tensors(self, sequence_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize only block-table metadata for a fused paged-attention kernel."""

        started = time.perf_counter()
        if not sequence_ids:
            raise ValueError("sequence_ids cannot be empty")
        sequences = [self.sequences[sequence_id] for sequence_id in sequence_ids]
        max_blocks = max(len(sequence.block_table) for sequence in sequences)
        block_tables = torch.full(
            (len(sequences), max_blocks),
            -1,
            device=self.device,
            dtype=torch.int32,
        )
        for row, sequence in enumerate(sequences):
            block_tables[row, : len(sequence.block_table)] = torch.tensor(
                sequence.block_table, device=self.device, dtype=torch.int32
            )
        lengths = torch.tensor(
            [sequence.length for sequence in sequences], device=self.device, dtype=torch.int32
        )
        self.metadata_rebuild_ms += (time.perf_counter() - started) * 1000
        return block_tables, lengths

    def free(self, sequence_id: str) -> None:
        sequence = self.sequences.pop(sequence_id)
        for block in sequence.block_table:
            self.allocator.free(block, sequence_id)
        self.device_block_tables[sequence.metadata_row].fill_(-1)
        self.device_context_lengths[sequence.metadata_row] = 0
        self._free_metadata_rows.append(sequence.metadata_row)

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
