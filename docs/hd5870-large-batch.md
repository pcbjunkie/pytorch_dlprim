# HD5870 larger training batches, 2026-10-29

Headless Cypress XT, 1 GiB VRAM, Phenom II X4 955/16 GiB. Same fixed YOLO nano five-class checkpoint as the utilization study, 256px FP32, automatic workgroup cap 32 / BN 64, queue 64 and per-kernel submission flush enabled. One loader worker, fused EMA, no profiling or CPU checks in native timing. Each accepted candidate passed two separate CPU loss/full-gradient comparisons before a fresh native process measured 16 updates after 6 warmups. No checkpoint/model changes saved.

| Batch | Requested reserve MiB | Images/s | Mean batch ms | Backend peak reserved MiB | GPU mean / peak | Whole-card VRAM peak MiB | GTT peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 768 | 6.291 | 1907.4 | 437.0 | 43.0% / 79.2% | 783.5 | 2.5 |
| 16 | 768 | 8.656 | 1848.4 | 572.6 | 56.6% / 82.5% | 784.8 | 2.6 |
| 20 | 768 | 9.805 | 2039.7 | 706.6 | 61.1% / 88.3% | 785.0 | 2.5 |
| 24 | 960 | 10.371 | 2314.2 | 841.7 | 63.6% / 85.8% | 977.9 | 2.6 |

The previous batch 8 / reserve 512 probe measured 4.998 images/s. The changed reservation must be retained when comparing results. Backend peak bytes omit the full driver reservation and other allocations. Whole-card VRAM includes that pool; it is not a tensor-live-bytes counter. VRAM/GTT peaks can occur at different times and must not be added. Monitor samples are one second apart.

## Failed or limited candidates

No candidate process exited on an allocation/resource error.

## Throughput interpretation

Batch 24 was fastest at 10.371 images/s. Batch 24 / reserve 960 improves about 5.8% over batch 20 / reserve 768 and about 107.5% over the earlier batch 8 / reserve 512 result. Gains continue beyond 20, with diminishing returns. Batch 20 uses the smaller reservation for a modest throughput cost. This curve does not establish a purely compute-limited bottleneck or a sustained-training recommendation.

## Kernel messages

The batch 20 CPU-reference process logged TTM placement/eviction failures for a 768 MiB buffer. Both CPU loss/gradient comparisons still passed and the process completed. These warnings are retained rather than labeled as a clean kernel run. No GPU reset/lockup/stall was observed. The native throughput processes did not add kernel messages. Requested initial pool reservations produced no fallback messages; whole-card VRAM samples were consistent with the larger reservations.

## Scope

These are short throughput/capacity probes. They do not prove sustained epoch stability or an exact maximum batch size. The earlier HD6850 also passed batch 24 only in short trials; its full 20-epoch stage used batch 20 / reserve 768. Larger workgroup limits remain unchanged and untested here.

The active root follow-new kernel capture contained 89060 bytes at report time. Kernel excerpts are retained in the JSON and local per-stage logs.

[Machine-readable results](benchmarks/hd5870-large-batch-20261029.json), [previous training/inference profiles](hd5870-utilization.md), [initial card results](hd5870-results.md). Raw local logs/controller: `runs/hd5870-large-batch-20261029/`.
