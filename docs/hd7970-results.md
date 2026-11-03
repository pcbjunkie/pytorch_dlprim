# HD7970 baseline results

Sapphire Dual-X, Tahiti XT, PCI1002:6798,3GiB VRAM. Headless PhenomIIX4955,
16GiB RAM, Debian12/Linux6.1.0-42, Mesa25.0.7 Clover/radeonsi. Stock
Ultralytics8.3.0, Torch2.4.1 without AVX requirements,256px FP32.

**Power qualification:** user reports two8-pin GPU sockets and a6-pin PSU plug
in an8-pin socket. Exact remaining PSU-plug layout and boardSKU/BIOS are not
confirmed. User chose to continue. Results apply to this recorded setup;
they do not establish normal connector provisioning, board wattage, or absence
of intermittent throttling. Core clock reached the reported925MHz maximum.

## Completed stability run

- 20addedepochs (21–40), batch80, one loader worker, exit0.
- Training averaged20.590images/s; finalfive averaged20.558images/s.
- Elapsed6415.58seconds (1h46m56s), including validation/checkpoints/finalCPUevaluation.
- Savedbest checkpoint reloaded and CPUvalidated; this is separate fromGPUinference timing.
- Peak backend-reserved2818098820bytes (2687.55MiB).
- Whole-card stability peaks: VRAM3060.59MiB,
 GTT1494.96MiB, sampled every second.
- No captured GPU reset, lockup, ringstall or allocation warning. One unrelated
 cupsd AppArmor audit event was recorded and retained.

## Batch sweep

No r600 pool preallocation. Fresh processes, six warm-up and24measured updates;
no profiler orCPUreference comparisons inside measured windows.

| Batch | Images/s | Peak backend-reserved MiB |
| ---: | ---: | ---: |
| 12 | 6.367 | 436.37 |
| 20 | 11.129 | 706.62 |
| 24 | 11.609 | 841.77 |
| 28 | 13.675 | 977.33 |
| 32 | 15.213 | 1112.01 |
| 40 | 16.519 | 1382.22 |
| 48 | 18.705 | 1652.45 |
| 56 | 19.270 | 1922.70 |
| 64 | 19.733 | 2192.93 |
| 80 | 20.057 | 2687.52 |
| 96 | 17.937 | 3218.82 |
| 112 | 15.418 | 3750.11 |
| 128 | 14.580 | 4281.41 |

Batch80 was fastest. Batches2/24/80 passed two full-modelCPUloss/gradient
comparisons each. Larger batches completed but slowed; allocation above3GiB
is not proof of VRAM residency or an exact maximum batch capacity.

## Inference

Same frozen model and20-image manifest as the other cards, batch1, fusedFP32,
20warm-ups and100timedframes per mode. All800CPU/raw-output/NMS comparisons
passed outside timing. Independent windows have an unusual ordering; retain
these numbers and investigate in phase two instead of inferring a benefit
from decoding.

| Mode | FPS | Mean ms | P95 ms |
| --- | ---: | ---: | ---: |
| device resident forward | 9.599 | 104.18 | 110.58 |
| decoded frame to detections | 11.536 | 86.68 | 118.51 |
| encoded image to detections | 13.915 | 71.87 | 109.52 |

## Software and retained evidence

Same V9 binary as the completedR9pass, SHA256
`5b25148c3f1d093a9f8b21ec8207e9fb5910cb9cf5e1f7cd02515310a8a46ed9`.
No-cache/tight allocation, deferred event retirement, queue128/budget64MiB.
No r600 shader shim or reservation. Ordinary-cache R9 correctness failures are
preserved separately; these results do not establish that path is fixed.

The allocation gate held2880MiB live in45buffers and fully verifiedGPUupdates
and hostreadback; counters returned to zero after release. Sampled peaks were
2900.05MiB VRAM/69.64MiB GTT. Low GPUbusy observations and memory residency
changes are recorded for phase-two profiling. Busy% is not ALUoccupancy or a
linear speedup estimate; no readable board-power sensor was available.

Raw workspace evidence: `runs/hd7970-baseline-v9-retired-20261030/`,
`runs/hd7970-allocation-20261030/` and
`runs/hd7970-wildlife-v9-retired-stability20-20261030/`.
An intentional interruptedbatch56 trial is archived and was rerun; completed
measurements and the connector qualification are retained. No additional GPU
experiments were launched during final review.
