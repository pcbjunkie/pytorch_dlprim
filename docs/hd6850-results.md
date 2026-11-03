# HD6850 results, 2026-10-29

Sapphire HD6850 / Barts PRO, PCI `1002:6739`, subsystem `174b:e174`, 1 GiB
VRAM; Phenom II X4 955, 16 GiB RAM. Early trials used XFCE; later
headless trials are identified below. Debian 12/Linux
6.1.0-42-amd64, LLVM 15.0.6, isolated patched Mesa 25.0.7 Clover/r600,
Torch 2.4.1 local no-AVX CPU wheel and Ultralytics 8.3.0.

These results use the local `pool32-v1-20261028` build, including the newer
`src/scatter_ops.cpp` and `src/training_ops.cpp` fixes. The published backend
base `3cd96dc9` alone does not include those post-snapshot fixes. Binary/source
hashes and exact results are recorded in the
[machine-readable card record](benchmarks/hd6850-training-reservations-20261029.json).

## Sustained and integration evidence

- Seven full fine-tuning epochs completed at batch 2, 256px FP32, one loader
  worker, 256 MiB initial pool reservation and pending-operation limit 32.
  The user requested stopping after epoch 7 validation/checkpoint; epoch 8
  was not started. Setup, training, validation and saving took about 5h16m.
- All 86 periodic CPU loss comparisons passed, across 21,420 updates;
  maximum loss difference was 0.00001526. Epoch validation losses remained
  finite. This sustained run did not compare every gradient against CPU.
- Starting weights were the final HD6970 30-epoch checkpoint; this was a new
  SGD/EMA/schedule stage, LR0 0.001 and LRF 0.1. Final epoch-7 validation
  mAP50/mAP50–95 was 0.49070/0.26991. The final standalone CPU validation
  normally performed at trainer shutdown was not run at the requested stop.
- A separate two-epoch, small real-image integration test passed nine CPU
  loss/gradient/SGD/EMA comparisons, GPU and CPU validation, checkpoint
  saving/reload and a resumed update with restored optimizer/EMA state.
  The new pooling regression passed 59 CPU comparisons.

The batch/reservation sweeps below are short trials. A later 20-epoch headless
batch-20 stage is recorded separately. Kernel-capture limitations are retained
with each result; training completion alone is not a complete reset audit.

## Unprofiled throughput

YOLOv8 nano detection, five wildlife classes, 6,119 training/707 validation
images, 256px FP32, one loader worker and two CPU threads. All short trials
below start from the same saved epoch-7 checkpoint. Six warm-up updates precede
the measured window. Loading/augmentation, transfers, forward/loss, backward,
clipping/SGD, EMA and normal progress logging remain included.

**Operator profiling and CPU reference comparisons are absent from these timed
windows.** External radeontop samples whole-card usage once per second.

| Batch | Initial reserve MiB | Pending-op limit | Measured updates | Images/s | Mean / peak GPU % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2, repeated control | 256 | 32 | 40 | 2.126 | 39.9 / 53.3 |
| 12 | 512 | 32 | 20 | 5.843 | 64.0 / 90.8 |
| 12, confirmed | 512 | 64 | 40 | 6.376 | 69.4 / 90.0 |
| 16 | 640 | 64 | 20 | 6.929 | 75.4 / 93.3 |
| 16 | 768 | 64 | 20 | 6.905 | 76.1 / 95.0 |
| 16 | 960 | 64 | 20 | 6.207 | 72.7 / 94.2 |
| 20 | 960 | 64 | 20 | 6.730 | 74.2 / 96.7 |
| 24 | 960 | 64 | 20 | 6.518 | 77.7 / 95.0 |

Batch12/queue 64 confirmed about **3.00x** the repeated batch-2 control. Its
queue 64 comparison improved about 7.3% over queue 32. These pending-operation
limits are not threads per workgroup; workgroup compatibility settings stayed
unchanged. Batch16/reserve 640 is the fastest short candidate; its small
difference from reserve 768 is within plausible run variation. Throughput
plateaus around 6.5–7 images/s as batches grow further.

Separate full-model CPU loss/gradient checks pass at batch 12/reserve 512,
batch 16/reserve 640 and batch 24/reserve 960, all at queue 64. At batch 24 the
maximum loss difference was 0.00003052 and minimum gradient cosine exceeded
0.999999999997. CPU checks download weights/gradients and run CPU mathematics;
their lower/jumpier GPU utilization is not the native throughput pattern.

The earlier R9 270 and HD6970 results used different backend snapshots,
checkpoints and batch configurations. These numbers are not a controlled
hardware-only speed comparison.

## Allocation and capacity evidence

| Configuration | Observation |
| --- | --- |
| No initial reservation | Short batches1/2/4/6 pass; batch 8 fails allocation |
| Reserve256 | Batch6 passes; batch 8 fails while requesting a larger pool |
| Reserve512 | Batch12 passes; batch 16 fails around a 513 MiB replacement request |
| Fresh reserve 640/768/960 | Batch16 completes; reserve 640 has separate CPU checks |
| Fresh reserve 960 | Batches20/24 complete; batch 24 CPU checks pass |
| Batch28/reserve 960 | `Not enough memory for command submission`; stopped there |
| Fresh reserve 1024 | Initial request fails and falls back to 512; not counted as a 1024 MiB pass |

Fresh640/768/896/960 MiB reservations also passed a tiny shader/readback test.
That test binds the driver pool but does not scan its entire capacity. The
reported maximum individual application buffer is 256 MiB; an internal driver
pool is a different allocation from one application `cl_mem`.

This disproves a general **512 MiB initial-allocation ceiling**. The old/new
pool-growth situation matters; it does not prove a uniquely GTT-caused limit.
The largest initial reservation tested successfully is 960 MiB, not a proven
exact maximum. Batch24 is the largest CPU-checked short batch;25–27 and 32
were not tested after the batch 28 failure.

At batch 24 the backend peak was 883,140,136 bytes (about 842 MiB). Whole-card
VRAM near1013 MiB includes the960 MiB reservation, XFCE and other allocations;
it is not the live tensor count. VRAM/GTT peaks may occur at different times
and must not be added. Migration/eviction was not traced.

Neither VRAM size nor one successful reservation determines a reliable setting
for another card. A later batch-20/reserve-768 stage completed 20 epochs; the short batch-24
capacity result still lacks equivalent sustained-duration evidence. No allocator redesign was performed in this pass.

## Profile evidence

The completed batch 12/reserve 512/queue 64 diagnostic profile inserted phase
barriers and enabled queue profiling; its timings are not an unprofiled speed
benchmark:

| Phase | Milliseconds/batch 12 |
| --- | ---: |
| Image loading | 4.4 |
| Preprocessing/input transfer | 8.1 |
| Network forward | 364.3 |
| Detection loss | 360.3 |
| Backward | 1085.9 |
| Gradient clipping | 38.3 |
| SGD | 78.7 |
| EMA | 47.1 |
| Complete batch | 1999.8 |

Backward convolution input/filter gradients dominate GPU event time. Loading
is about 0.22% of this diagnostic batch; a second worker was not tested because
the measured loader wait was small. Guarded/native batch 2 probes were nearly
tied (2.131/2.120 images/s), so those measurements do not identify the scalar
correctness guards as a meaningful throughput cost.

A fresh batch 20/reserve 960/queue 64 profile completed afterward. It used
four warm-up updates, ten natural-boundary batches, five synchronized phase
batches, three event-instrumented batches and two CPU-trace batches:

| Phase | Milliseconds/batch20, synchronized |
| --- | ---: |
| Image loading | 4.3 |
| Preprocessing/input transfer | 317.1 |
| Network forward | 599.9 |
| Detection loss | 440.0 |
| Backward | 1861.4 |
| Gradient clipping | 39.2 |
| SGD | 73.4 |
| EMA | 35.5 |
| Complete batch | 3378.7 |

Preprocessing/input transfer took about 240 ms at natural boundaries and
317 ms with phase barriers. That is an investigation target; input transfer
and normalization are grouped together, so this does not isolate the exact
copy or prove eviction. Filter partials and convolution input gradients remain
the largest GPU event groups. Loading images is still a small fraction.

The separate unprofiled batch 20/reserve 960 control measured 6.999 images/s
over forty updates. Do not substitute the instrumented phase timings for
either that control or the earlier 6.730 images/s twenty-update trial.
A matched headless comparison then completed, with forty measured updates
after six warm-up updates:

| Display state | Batch | Reserve MiB | Images/s |
| --- | ---: | ---: | ---: |
| XFCE active, new control | 20 | 960 | 6.999 |
| Headless | 20 | 960 | 7.281 |
| Headless | 20 | 768 | 7.298 |

The headless reserve sizes are essentially tied. Against the new desktop-on
control, the headless 960 MiB trial improved about 4.0%. Separate CPU
loss/gradient comparisons pass at headless batch 20/reserve 768. These remain
short trials, with no sustained headless/full-validation claim.

The selected headless reserve 768 profile measured 5.7 ms input
preparation/transfer at natural boundaries and 7.7 ms with phase barriers,
versus 240–317 ms in the earlier desktop-on reserve 960 profile. Both display
state and reservation changed between those two profiles; this is not an
exclusive eviction or driver-root-cause attribution. The synchronized headless
batch took 2.891 s: forward 552 ms, loss 478 ms, backward 1669 ms, clipping
37 ms, SGD 78 ms and EMA 53 ms; loading took 3.8 ms.

Per instrumented batch, convolution input-gradient kernels total about 659 ms
and filter-gradient kernels total about 552 ms. The final filter partial merge
accounts for only 1.9 ms of the latter. The remaining major costs are the
convolution computations, rather than image-loader starvation. A separate
existing filter-partition-target check follows; it does not change local
workgroup sizes or add new kernels.

## Larger workgroup follow-up

Headless checks passed automatic workgroup caps 64, 128 and 256, with
pointwise-output-offset, packed-byte, scatter and pooling CPU regressions.
Separate BN caps 64, 128 and 256 passed ten module cases each, including
two-stage statistics/gradient reductions. The exact previously captured BN
kernel also passed three CPU-referenced submissions at each size.

Two full-model CPU loss/gradient comparisons passed for each automatic 64/128
candidate, each BN128/256 candidate, and the combined automatic256/BN256
candidate. These caps permit larger groups where applicable; explicit kernel
sizes and the other indexing/GEMM/convolution compatibility settings remain.
The device advertises 256 as its maximum workgroup size.

The final matched probes use batch20, reserve768, queue64, one worker,
six warm-up and forty measured updates, with no operator profiling or CPU
comparisons in the timed window:

| Automatic cap | BN cap | Images/s |
| ---: | ---: | ---: |
| 32, fresh control | 64 | 7.282 |
| 256, combined candidate | 256 | 7.340 |

The measured difference is +0.78%; these short trials do not establish
long-term stability. There were no new GPU resets/lockups in the live kernel
audit; allocation/eviction warnings occurred during operator regressions.

The current HD6850 passing the old reproducer does **not** establish that
Cayman is inherently affected or that the old issue is fixed. Retesting the
HD6970 with the same current stack is required to distinguish later fixes
from device differences. Generic legacy defaults remain unchanged.

The separate filter-partition-target128/256 experiment was effectively tied
(ratio1.0000367), so target256 was retained.

## Completed WG256 twenty-epoch stage

The additional fine-tuning stage completed 20 epochs / 6,120 training updates
at batch 20, reserve 768 MiB, queue 64, one worker, headless, 256px FP32,with
both automatic and BN caps 256. All epoch loss/metric CSV values were finite.
No operator profiling or CPU comparison/debug callbacks ran during training.
Elapsed time including validation/checkpoints/final CPU evaluation was 5.005 hours;
training-only throughput averaged7.404images/s and settled
near 7.53 images/s in the final epochs. The best saved checkpoint reloaded and
completed full CPU validation, agreeing with final GPU metrics at printed precision.

Starting epoch 7 to final epoch 20 mAP50 improved from 0.49070 to 0.68226 and
mAP50–95 improved from 0.26991 to 0.40885. This was a fresh SGD/EMA/scheduler stage
initialized from the saved weights, not continuation of the old optimizer state.

The saved live capture missed a later kernel warning. A user-provided dmesg
excerpt at uptime 40174.721560 confirms a TTM placement/eviction failure for a
768 MiB buffer about 14m41s into training, near the first validation. The tables
show about 222 MiB free in VRAM and 254 MiB in GTT. The exact triggering
operation is unproven. No reset/lockup is shown in this excerpt; the user reports
no subsequent kernel entries. All 20 epochs and final CPU validation completed.
The stopped root tail prevents treating the saved capture as a full reset audit.

## Batch-one inference baseline

The final 20-epoch model was benchmarked on 20 fixed SHA-checked validation
images covering all five classes, 256×256 FP32, fused/eval, headless. Both GPU
trials use the conservative automatic cap 32, queue 64 and two CPU threads.
Each passed 400 raw-output/NMS CPU comparisons before timing and 400 afterward.
Twenty warmups precede 100 measured samples per mode:

| Device/configuration | Pipeline | Mean ms | Median ms | P95 ms | FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| Phenom CPU, 2 threads | Prepared-input forward | 115.35 | 115.28 | 117.30 | 8.67 |
| Phenom CPU, 2 threads | Decoded BGR frame → detections | 117.81 | 117.72 | 119.02 | 8.49 |
| Phenom CPU, 2 threads | Encoded image → detections | 123.14 | 122.36 | 131.58 | 8.12 |
| HD6850, reserve 128 MiB | Prepared-input forward | 57.72 | 54.97 | 72.84 | 17.32 |
| HD6850, reserve 128 MiB | Decoded BGR frame → detections | 64.45 | 61.66 | 77.37 | 15.52 |
| HD6850, reserve 128 MiB | Encoded image → detections | 69.51 | 67.51 | 80.62 | 14.39 |
| HD6850, reserve 768 MiB | Prepared-input forward | 59.57 | 57.20 | 69.56 | 16.79 |
| HD6850, reserve 768 MiB | Decoded BGR frame → detections | 68.96 | 66.23 | 82.70 | 14.50 |
| HD6850, reserve 768 MiB | Encoded image → detections | 73.62 | 70.66 | 88.69 | 13.58 |

The 128 MiB reservation is the selected inference candidate: it reserves much
less driver memory and was faster in these trials. Peak backend-reserved storage
was about 31 MiB, which is not whole-card VRAM usage. All measured processing
samples at that setting completed within 100 ms. Camera capture, video decoding,
HTTP, drawing, output encoding and filesystem reads are outside the measurements.

Repeated fused inference exposed intermittent raw-prediction differences at
automatic caps 64/128/256, despite passing shorter checks and the completed
training stage. Failure input/output tensors were retained; tolerances were not
relaxed. Cap 32 passed 2,000 strict cross-device comparisons across a separate
400-frame probe and the final trials. The exact failing kernel/compiler cause
is unresolved. Keep inference on the conservative automatic cap 32.

The first benchmark controller disappeared after 60 checks while the host boot
survived the reported power interruption. Partial logs were retained, and detached
reruns completed both GPU trials. Root dmesg capture had stopped, so no complete
kernel reset audit is claimed. All model/checkpoint files were preserved.

[Repeat this inference protocol](inference-benchmarks.md) on every subsequent
card with the same checkpoint, image manifest and measurement settings.

## Driver and headless access

A separate quick Mesa 25.0.7 Rusticl/r600 test passed host transfers but failed
shader readback. That is not a Rusticl training-support result; Clover passed
the same small test. Newer Rusticl versions were not tested in that comparison.

On this Debian host `/dev/dri/renderD128` belongs to the `render` group.
Stopping GDM removed the graphical-seat ACL and initially denied GPU access
to fresh processes. Persistent render-node permission and fresh group
membership are required for a headless account; merely belonging to `video`
was insufficient here. That permission-only startup failure is separate from
kernel, compiler and allocation failures.

## Reproduce a bounded throughput probe

Build the current source with the required HD6850 fixes and follow the
[isolated legacy profile setup](r600-clover.md). Supply the same dataset and
checkpoint explicitly; neither is distributed here. From the backend root:

```sh
export OCL_ICD_VENDORS=/path/to/isolated-clover-icd
export R600_MESA_PREFIX=/path/to/patched-mesa
R600_POOL_RESERVE_MB=960 OPENCL_ASYNC_OP_LIMIT=64 \
  tools/with-r600-clover.sh python examples/train_yolo_opencl.py \
  --data /path/to/data.yaml --model /path/to/epoch7.pt \
  --imgsz 256 --batch 20 --workers 1 --epochs 1 --lr0 0.001 \
  --warmup-steps 6 --probe-steps 20 --project /path/to/probes --name hd6850-b20
```

The portable adapter provides the same bounded trainer protocol; the original
measurements additionally used a local two-timestamp/final-counter wrapper.
Small numerical/operator regressions and separate CPU comparisons precede any
new sustained-training recommendation.
