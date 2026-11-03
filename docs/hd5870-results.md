# HD5870 bring-up and inference, 2026-10-29

Cypress XT, PCI `1002:6898`, subsystem `1002:0b00`, 1 GiB VRAM; Phenom II X4
955, 16 GiB RAM, headless, Linux 6.1.0-42, patched Mesa 25.0.7 Clover and
LLVM 15.0.6. Same frozen backend/driver libraries as the HD6850, with an extra
opt-in command-submission shim for fused inference. The PCI subsystem description says Eyefinity
Edition; the observed memory capacity is 1 GiB.

## Correctness evidence

Ten selected current-profile operator regressions passed, covering copies,
offsets, unique/top-k/masks, index reductions/scatter, pooling, SiLU backward,
deferred lifetimes and memory reporting. The small filter-gradient regression
also passed. Two full YOLO training updates passed separate CPU loss/gradient
checks: loss errors below 0.000002 and gradient cosine above 0.999999999997.
The first full-model iteration incurred substantial cold kernel compilation.
This is bounded training compatibility evidence, not a sustained-training claim.
The subsequent [training/inference utilization study](hd5870-utilization.md)
adds separate CPU-checked batch sweeps and phase/event/API profiles.
The [larger training-batch follow-up](hd5870-large-batch.md) extends those probes
using 768/960 MiB initial reservations.
All tested batches through 24 passed the short CPU-gated native probes; batch 24
reached 10.37 images/s with a 960 MiB reservation. This remains short-run evidence.
The subsequent [backward-pass follow-up](hd5870-backward.md) adds a narrow
input-gradient tile and specialized depthwise filter selection/indexing fixes.
At batch 20 / reserve 768, the combined candidate measured 10.76 images/s
versus an original control mean of 10.30; these are matched short probes.
The [synchronization/utilization audit](hd5870-synchronization.md) found the
GPU generally active during host blocking calls; deferred marker-flush batching
passed its correctness gates but showed no throughput gain, so that study
retained the V4 path.
The subsequent [parallel top-k study](hd5870-topk.md) adds the cooperative
selector and output-view/legacy-serial fixes. At batch 20 the matched short
native comparison is 11.566 versus a control mean of 11.080 images/s (+4.38%).
The updated batch-24 probes pass CPU checks and reach 12.00–12.06 images/s with
a 960 MiB reservation. This is short-run evidence; V8 is selected for the local
training wrapper while other cards await their own checks.

## Matched batch-one inference

Same final HD6850 20-epoch five-class model and frozen 20-image manifest as
the preceding card. Batch 1, 256×256 FP32, CPU convolution/BN fusion, two Torch
threads, one OpenCV thread, automatic workgroup cap 32, queue 64, initial pool
reservation 128 MiB. The passing Cypress run additionally enables
`R600_FLUSH_AFTER_KERNEL=1`, described below. The generic benchmark script is byte-identical across cards.
Each mode has 20 warmups and 100 timed frames. The GPU trial passed 400 raw/NMS
CPU comparisons before timing and 400 after; CPU comparisons are untimed.

| Processing | HD5870 mean ms | Median ms | P95 ms | HD5870 FPS | Phenom CPU FPS | HD6850 FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prepared-input forward | 55.15 | 53.42 | 72.33 | 18.13 | 9.22 | 17.32 |
| Decoded frame → detections | 57.36 | 52.15 | 86.06 | 17.43 | 9.17 | 15.52 |
| Encoded image → detections | 63.21 | 61.69 | 76.12 | 15.82 | 8.31 | 14.39 |

Frame processing includes preprocessing, tensor upload/normalization, network,
output readback, CPU NMS, scaling and detection dictionaries. The encoded-image
mode also decodes preloaded bytes. Camera/video capture, HTTP, drawing, output
encoding and filesystem reads are excluded. Memory counters describe backend
storage, not whole-card VRAM. CPU control measurements and boots differ; do not
attribute every cross-card timing difference to GPU hardware alone.

The fresh root kernel capture remained active and empty throughout these checks:
no allocation warnings, GPU resets or lockups were logged in this test window.
The later utilization study tested batches 1/2/4/8, a 512 MiB training reservation,
and separate training/inference profiles. The larger-batch follow-up passed
12/16/20/24 with 768/960 MiB reservations. It logged recovered TTM placement
warnings during the batch-20 CPU-reference checks; native speed tests added no
kernel messages. Neither study established exact maximum batch capacity or long
training stability. Larger workgroup caps from the previous card were not
carried over. Use this conservative workgroup profile initially.

## Queued-inference correction

The original cap-32, queue-64 fused-inference run failed the first raw CPU
comparison: maximum decoded box error 1.75998 pixels and score error 0.00404836.
It produced no accepted timing result. Failure tensors and logs were preserved.
A layer trace found incorrect concatenated inputs before the first differing
convolution; the exact corrupting operation is not isolated. Retaining model
intermediates did not fix the problem. A smaller producer/slices/concat test
passed, so it is not an independent reproducer of the full-model failure.

A synchronous queue-limit-1 control passed 400 predictions. Calling `clFlush`
after each successful kernel enqueue also passed 400, without inserting
`clFinish`. The opt-in submission shim preserves ordinary kernel arithmetic
and CPU-check tolerances; the passing final benchmark performs another 800
strict raw/NMS comparisons. This is a tested workaround, not a proven fix to
a specific Mesa source line or a general guarantee for other GPUs.

Build `tools/build-r600-flush-shim.sh`, then set `R600_FLUSH_AFTER_KERNEL=1`
before invoking `tools/with-r600-clover.sh`. The generic profile default remains
unchanged when the flag is absent. The local `with-hd5870-inference.sh` selects
this flag and the 128 MiB candidate reservation. Keep the original failure
evidence alongside the passing results.

The [machine-readable results](benchmarks/hd5870-bringup-inference-20261029.json)
retain hashes, raw timing samples, operator results, CPU gradients and limits.
Follow [the same inference protocol](inference-benchmarks.md) on every later card.
