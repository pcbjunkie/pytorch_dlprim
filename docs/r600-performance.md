# HD6970 performance evidence, 2026-10-28

These results are bounded measurements on one Cayman XT 2 GiB, Phenom II X4 955,
16 GiB RAM, Debian 12/Linux 6.1, LLVM 15 and isolated patched Mesa 25.0.7 Clover.
Torch is the local no-AVX 2.4.1 CPU build; Ultralytics is pinned to 8.3.0.
They are not a prediction for every VLIW GPU or 1 GiB configuration.

## Final filter partition and queue tuning

Same frozen candidate binary and fixed trained checkpoint, YOLOv8 nano detection,
five classes, batch 24, 256px FP32, one image worker, two CPU threads, 960 MiB
initial reserve and 32 MiB retirement threshold. Four warm-up updates precede
24 measured updates. Timed windows exclude operator profiling, CPU comparisons,
validation and checkpoint writes; normal image preparation, SGD and EMA remain.

| Configuration | Images/s | Peak backend reserved bytes |
| --- | ---: | ---: |
| Filter target 128, queue 16, control A | 11.80685 | 882,624,040 |
| Filter target 256, queue 16 | 12.39783 | 882,624,040 |
| Filter target 512, queue 16 | 12.20259 | 882,624,040 |
| Filter target 128, queue 16, control B | 11.86621 | 882,624,040 |
| Filter target 256, queue 32, later comparison | 12.75619 | 882,624,040 |
| Filter target 256, queue 16, later comparison | 12.23598 | 882,624,040 |

Target 256 improved throughput **4.74%** against the mean of the bracketing
controls. Queue 32 improved **4.25%** against its later queue-16 comparison.
The selected 256/32 result is **7.77%** above the original control mean. Do not
multiply the individual percentages: windows differ and these are short runs,
not confidence intervals. Peak backend storage was unchanged in all six probes.

The partition target increases independent reduction groups while retaining
32x32 output tiles, K32 and 64-thread workgroups. The maximum is still 64
partitions, at least 512 reduction rows per partition, and 4 MiB scratch.
Default non-profile selection stays at target 128. No larger local workgroup,
new atomic operation or new tile-depth algorithm was introduced.

Exploratory raw OpenCL tile-depth/partition probes showed intermittent CPU
discrepancies, including an unchanged control. An initial one-partition harness
layout error was corrected separately. These raw probes are not evidence of
general kernel correctness; the integrated checks below are separate evidence.
The standalone discrepancy's root cause remains unresolved; tile-depth changes
were rejected rather than shipping a new unvalidated variant.

## CPU and integration evidence

- Target-256 convolution regression: seven cases, repeated three times, including
  groups, dilation, last smaller batch, a reduction tail and leaf accumulation.
- Three full-model CPU loss/gradient comparisons at queue 16: minimum cosine
  0.99999556; maximum loss discrepancy 0.00003052; norm ratios within 0.99..1.01.
- Two more full-model comparisons at queue 32: minimum cosine 0.99999978;
  maximum loss discrepancy 0.00002289; norm ratios within 0.99..1.01.
- Queue-32 temporary churn, aliasing, bounded retirement and accounting checks
  passed; live/reserved/deferred counters returned to baseline after release.
- All 10 checks in the selected operator suite passed on the final profile;
  the distributed runtime patch reconstructs all modified/new files and builds
  successfully from a pristine upstream archive. The portable example completed
  a short train/validate/save/reload cycle.
- Memory-reporting checks cover known allocation size, aliases, release, peaks,
  resets, decimal GB and unavailable driver counters. Progress reads do not add
  a blocking queue synchronization.

The completed 30-epoch run used the preceding target-128 / queue-16 snapshot,
not this final tuning. It averaged **12.297 images/s** in training-only epoch
timings (validation/checkpoint time excluded), with finite losses and CPU/GPU
validation agreement. It established a useful stable baseline. Final tuning
still needs an equivalent sustained run before claiming the same duration.

## Fresh profile

Five synchronized phase batches, with queue profiling enabled:

| Phase | Seconds/batch |
| --- | ---: |
| Network forward | 0.38535 |
| Detection loss | 0.40432 |
| Backward | 1.03161 |
| Gradient clipping | 0.03484 |
| SGD | 0.06939 |
| EMA | 0.04808 |
| Complete batch | 1.99627 |

Three event-instrumented batches give 319.22 ms of filter partials, 7.23 ms
remaining filter GEMM and 1.09 ms final filter reduction. The preceding profile
measured 336.00, 40.35 and 0.68 ms respectively: roughly **13.1% less GPU time**
for this filter-kernel set. Input gradients remained about 283.59 ms and forward
GEMM 239.64 ms. Weight gradients remain the largest math target; this tuning did
not remove all remaining bottlenecks. Phase wall time and GPU event time must
not be added together. Profile timings include instrumentation overhead.

## Earlier progress and reproducibility

Earlier matched passes measured +8.3% for power-of-two input-gradient indexing,
+18.3% for atomic-free split filter gradients, +12.1% for fused SiLU backward,
+5.0% for four-pixel 1x1 input gradients, +25.0% for bounded deferred release,
and +10.4% for four-pixel 3x3/small-1x1 gradients. These used different intermediate
snapshots/batches and are not independent percentages to multiply. Allocation,
pool reservation and indexed-reduction changes also enabled larger batches.
The old batch-8 baseline near 2.6 images/s and the final batch-24 result therefore
describe total system progress, not a pure same-batch kernel speedup.

Machine-readable [benchmark/profile data](benchmarks/hd6970-filter-tuning-20261028.json)
and [CPU comparisons](benchmarks/hd6970-filter-cpu-checks-20261028.json) retain
the exact settings, checkpoint hash and measured values. Runtime patches include
all kernel sources; [the build/profile guide](r600-clover.md) explains applying
them. `tests/local/run_r600_checks.py` is the selected repeatable operator suite.
An optional portable trainer/benchmark adapter is included under `examples`;
see its README. Dataset/model paths are explicit command arguments; no data,
model or private local path is part of this source snapshot. R9 270's earlier
~19.7 images/s used batch 32 and
a different backend/architecture, so it is not a matched hardware comparison.
