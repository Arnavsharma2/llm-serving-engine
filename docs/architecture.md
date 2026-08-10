# Architecture notes

## Cache layout

The allocation unit is a token block, not a request. One tensor owns all physical storage:

```text
[layer, K-or-V, physical_block, token_offset, kv_head, head_dim]
```

Each sequence stores only an ordered list of physical block ids and its logical length. Reserving a
token allocates a block only at block boundaries. Freeing or evicting a sequence returns every block
to a FIFO free list in O(number of owned blocks). The allocator retains ownership metadata so a
double free or cross-sequence free fails loudly.

For a model with `L` layers, `Hkv` KV heads, head dimension `D`, block size `B`, and element size `E`,
one physical block costs:

```text
2 * L * Hkv * D * B * E bytes
```

Contiguous preallocation reserves `max_sequence_length` slots per live sequence. Paged allocation
reserves `ceil(actual_length / B) * B`, so internal waste is bounded by `B - 1` tokens per sequence.
`PagedKVCache.stats()` reports both quantities at the paged-cache high-water mark.

## Paged attention

The serving attention path uses an online-softmax recurrence. For each physical block it computes
the block scores, updates the running maximum, rescales the previous accumulator, adds the current
weighted values, and updates the normalization term. Only a block of scores exists at once. This is
the same numerical decomposition a fused kernel would use, expressed as inspectable PyTorch.

## Request lifecycle

1. A request enters the incoming asynchronous queue.
2. The configured policy orders waiting requests.
3. At every model iteration, continuous mode fills open batch slots; static mode waits for the
   entire active batch to finish.
4. Every active sequence advances by one token. This permits mixed prefill, decode, and replay after
   preemption in the same iteration.
5. Cache pressure preempts the sequence with the most processed tokens, releases its blocks, and
   returns it to the waiting queue. On readmission, prompt plus already-emitted output tokens are
   replayed. Recomputation and preemption counts remain attached to the request.
6. A token callback drives both the benchmark timestamps and SSE output.

## Scheduler tradeoffs to measure

- FCFS provides the clearest arrival-order fairness but short requests can sit behind long ones.
- SJF reduces mean and often p95 latency for short jobs, but its output-length estimate can be wrong
  and sustained short arrivals can starve long requests.
- Priority scheduling is useful for service tiers but should always be reported with per-tier
  latency and starvation metrics.
