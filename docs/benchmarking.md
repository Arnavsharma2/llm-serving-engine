# Benchmark protocol

## Invariants

Use the same model revision, device, dtype, seed, arrival trace, prompt tokens, requested output
lengths, warm-up procedure, and stop-token behavior for every row. The harness creates the entire
trace before starting a configuration. It is open-loop: arrivals occur on schedule even when the
engine is overloaded, which makes queueing and tail latency visible.

The checked-in profile uses clipped log-normal distributions as a reproducible right-skewed proxy
for chat traffic. If an empirical ShareGPT trace is substituted, save the derived length sample and
its license alongside the run metadata.

## Metric definitions

- **TTFT:** first output-token timestamp minus client arrival timestamp. It includes queueing and
  prefill.
- **TPOT:** final-token timestamp minus first-token timestamp, divided by output tokens minus one.
- **End-to-end:** request completion minus arrival.
- **Throughput:** all successful output tokens divided by benchmark wall time.
- **GPU memory:** highest NVML used-memory sample during the run.
- **GPU utilization:** arithmetic mean of NVML GPU-utilization samples.

JSON contains the summary, workload configuration, and raw per-request rows. CSV contains the same
per-request measurements for independent analysis.

## Required experiment matrix

1. Naive versus paged at low arrival rate to isolate cache reuse.
2. Static versus continuous at matched maximum batch size under increasing arrival rate. Plot p95
   TTFT and e2e, not only peak throughput.
3. Continuous FCFS versus SJF. Report p50/p95/p99 by requested output-length quartile so an aggregate
   win cannot hide starvation.
4. Cache block sizes 8, 16, and 32. Plot internal fragmentation and throughput.
5. FP versus INT8 versus INT4 on the same WikiText token slice, HellaSwag examples, and decode
   prompts. Report both accuracy and throughput regressions.
6. Speculative decoding by prompt domain and offered load. Record acceptance rate and target passes;
   include the high-batch regime where draft cost can make it slower.

## Before publishing a number

- Run a warm-up excluded from measurement.
- Repeat at least five times and report median plus dispersion.
- Record GPU model, driver, CUDA, PyTorch, model revision, dtype, power limit, and command.
- Inspect raw failures and ensure successful-request filtering is disclosed.
- Never label the packed INT4 reference layer GPTQ/AWQ or claim kernel speedups it does not provide.
