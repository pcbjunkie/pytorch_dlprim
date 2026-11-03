# R9 270 benchmark and stability results

Completed baseline: YOLOv8 nano, five wildlife classes, 256px FP32, stock
Ultralytics 8.3.0. Host: Phenom II X4 955, 16 GiB RAM, headless Debian 12.
GPU: PITCAIRN, PCI 1002:6811, 2 GiB VRAM, system Mesa 25.0.7 Clover/radeonsi.
Torch is the local 2.4.1 build without AVX requirements.

## Training and stability

All 20 added epochs (21–40) completed with exit code 0, at batch 48 and one
loader worker. Total elapsed time was 2h23m08s, including validation, checkpoints
and final CPU evaluation. Training-only throughput averaged **15.289 images/s**;
the final five epochs averaged **15.386 images/s**.

The saved best checkpoint was reloaded and validated on CPU. Final-epoch
mAP50/mAP50–95 was 0.69602/0.42041. Final CPU validation timings are separate
from the GPU inference benchmark below. The stability kernel capture recorded
**zero new messages**: no captured allocation warning, GPU reset, lockup or
ring stall during this run.

Peak backend-reserved storage was 1,733,772,912 bytes (1653.46 MiB). There is
no r600-style pool reservation on this GCN card. Whole-card one-second samples
recorded peaks of **2029.34 MiB VRAM** and **789.68 MiB GTT**
across 8586 samples. These include the driver and validation work; they
are distinct from process allocator counters and do not prove every buffer
remained in VRAM.

## Clean batch sweep

Each fresh process used six warm-up and 24 measured updates. No operator
profiling or CPU comparisons ran inside these windows. Image preparation,
transfers, forward/loss/backward, SGD and EMA remain included.

| Batch | Images/s | Peak backend-reserved MiB |
| ---: | ---: | ---: |
| 12 | 7.327 | 436.58 |
| 20 | 10.433 | 706.62 |
| 24 | 11.974 | 841.74 |
| 28 | 12.893 | 976.85 |
| 32 | 12.709 | 1111.97 |
| 40 | 14.092 | 1359.29 |
| 48 | 14.658 | 1626.90 |
| 56 | 13.447 | 1890.59 |
| 64 | 13.053 | 2165.42 |
| 80 | 12.961 | 2697.38 |
| 96 | 12.475 | 3230.65 |

Batch 48 was fastest. It passed two full-model CPU loss/gradient update
comparisons, as did batches 2 and 24. Larger batches completed through 96,
but slowed. Backend storage exceeded 2 GiB from batch 64; successful allocation
therefore does not establish VRAM residency or an exact maximum batch size.
Timestamped VRAM/GTT samples and raw allocator counters are retained per trial.

## Inference

The frozen original checkpoint and 20-image manifest match the other card
benchmarks. Batch 1, fused evaluation, FP32, two Torch threads and one OpenCV
thread. Each mode has 20 warm-ups and 100 timed frames. All **800 CPU/raw-output/
NMS comparisons passed** outside timing.

| Mode | FPS | Mean ms | P95 ms |
| --- | ---: | ---: | ---: |
| device resident forward | 11.727 | 85.28 | 110.35 |
| decoded frame to detections | 12.559 | 79.63 | 117.62 |
| encoded image to detections | 13.337 | 74.98 | 87.94 |

These independent windows show encoded processing faster than decoded
processing. Preserve that unusual ordering as measurement variability; it does
not show that decoding improves performance. Inference and the historical
roughly 19 training images/s claim remain targets for a matched phase-two
comparison. Historical runs with different checkpoints and batches are not a
controlled backend comparison.

## Software and correctness qualifications

The original V8 parallel top-k failed on this card. The
[V9 local-storage fix](r9-topk-local-storage.md) passes 54 focused and 129 broader
operator cases. V9 with ordinary caching then failed batch-24 model parity;
a synchronous no-cache control passed. This baseline explicitly uses no-cache,
tight allocation and deferred event retirement, queue limit 128 and a 64 MiB
retirement budget. No r600 shader shim or preallocation is used. The exact
ordinary-cache defect remains unresolved; this configuration is a verified
workaround. Legacy-card results used frozen V8; their V9 regressions are pending.

V9 binary SHA256:
`5b25148c3f1d093a9f8b21ec8207e9fb5910cb9cf5e1f7cd02515310a8a46ed9`.

The separate initial allocator probe held 1920 MiB live and verified GPU
updates with full readback. That test used V8 with ordinary allocation, so its
conditions differ from this training baseline. Keep requested payload,
backend-reserved storage, and actual VRAM/GTT residency separate.

Raw workspace evidence: `runs/r9-270-baseline-v9-retired-20261030/` and
`runs/r9-270-wildlife-v9-retired-stability20-20261030/`. Earlier failed CPU gates
and captures remain in their original directories.
[Overall machine-readable record](benchmarks/radeon-baselines-20261030.json).

Baseline reviewed and ready for the HD 7970 comparison. No additional GPU
experiments were launched during this review.
