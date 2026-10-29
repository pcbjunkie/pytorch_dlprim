# Changes by layer

The ordinary path is **YOLO → PyTorch dispatcher/autograd → pytorch_dlprim →
DLPrimitives/OpenCL kernels → Mesa Clover/r600 → Radeon kernel driver → GPU**.
Some Torch operations are implemented directly by backend-generated OpenCL
kernels rather than going through a DLPrimitives core operator.

## PyTorch core

No OpenCL changes were made inside PyTorch itself. The extension uses PyTorch's
existing PrivateUse1/custom-device dispatch interface. The separately built
Phenom-compatible CPU Torch wheel is a host compatibility requirement, not the
OpenCL implementation. No Linux kernel or LLVM compiler changes are included.

## Torch backend: this repository

| Modules | Changes |
| --- | --- |
| `src/CLTensor.h`, `src/CLTensor.cpp`, `src/python_iface.cpp`, `python/pytorch_ocl/__init__.py` | Per-device discovery/accounting; tight allocation in no-cache mode; bounded deferred retirement behind GPU markers; explicit synchronize/empty-cache drains; live/reserved/cache/deferred/peak counters |
| `src/index_ops.cpp`, `src/training_ops.cpp`, `src/scatter_ops.cpp` | Legacy 32-bit indexing/nonzero metadata and branchless division; bounded unique/topk paths; packed integer outputs; safe indexed writes and sorted atomic-free FP32 accumulation |
| `src/pointwise_ops.cpp` | Fused ordinary FP32 SiLU backward, retaining fallback for unsupported/higher-order cases |
| `src/tensor_ops.cpp` | Repeat/view fallback corrections and legacy byte-copy/conversion handling |
| `src/foreach_norm_ops.cpp`, `src/batch_pointwise_ops.cpp`, `src/ema_ops.cpp` | Earlier batched clipping norms, SGD/foreach and EMA updates, reducing small launches and transfers |

These are changes to the **extension**, not a fork of PyTorch core. Observed
operator warnings were addressed with GPU implementations for supported cases;
unsupported paths retain their existing diagnostics/fallbacks.

## DLPrimitives: separate compute library

Delivered as `patches/dlprimitives-r600-clover.patch`, applied inside the upstream
submodule at the base recorded in `patches/r600-bases.json`.

| Modules inside DLPrimitives | Changes |
| --- | --- |
| `src/core/pointwise.cpp` | Reproduced general output-storage-offset bug: homogeneous stores now include each output's offset |
| `src/core/conv.cpp`, `src/kernels/conv_input_gather.cl` | Destination-owned atomic-free input gradients; power-of-two address optimization; four-pixel weight reuse for eligible 1x1/3x3 shapes |
| `src/core/conv.cpp`, `src/kernels/sgemm.cl`, `src/kernels/filter_split_reduce.cl` | Disjoint filter-gradient partials plus fixed-order reduction; bounded workspace and tunable partition-group target |
| `src/core/pool.cpp`, `src/kernels/pooling.cl` | Destination-owned max-pool backward, avoiding floating-point CAS |
| `src/core/bn.cpp`, `src/gemm.cpp`, convolution kernels | Opt-in tested BN reduction/workspace caps and smaller GEMM tile/launch choices |
| `src/core/util.cpp`, `src/kernels/copy_strided.cl` | Checked/flattened legacy strided-copy path with explicit small workgroups |
| `src/core/pointwise.cpp`, `src/core/util.cpp`, `src/kernels/pack_byte_output.cl`, interpolation/depthwise kernels | Packed byte-output and legacy kernel/launch compatibility changes |
| `src/program_cache.cpp` | Optional portable reductions for the earlier Rusticl experiments |

The output-offset fix is a reproduced DLPrimitives defect. Gather/vectorized
gradients, split reductions and tile choices are algorithm/compatibility changes.
Flattened copies, branchless helpers and packed stores are workarounds for
observed failures on the legacy stack; they do not prove a particular compiler
or hardware root cause. They were implemented here because this layer generates
the kernels and controls their geometry.

## Mesa: actual r600 driver fixes

These are **separate from DLPrimitives** and delivered independently as
`patches/mesa-25.0.7-r600-clover.patch`:

| Mesa module | Changes |
| --- | --- |
| `src/gallium/drivers/r600/compute_memory_pool.c` | Ordered live allocation list and first-fitting-hole placement, addressing reproduced overlapping pool allocations; optional upfront reservation and fallback/checking for initial allocation failure |
| `src/gallium/drivers/r600/r600_hw_context.c` | Opt-in cache flushing before relevant CP DMA copies, addressing reproduced stale/corrupted shared-pool copy results |

Upfront reservation is a driver memory-management feature/tuning setting; it
does not increase physical VRAM or repair every pool-growth OOM path. The
wrapper supplies its requested size, but the driver implements it. Cache-flush
and placement changes are not Torch operators. Clover is Mesa's OpenCL frontend;
the patched allocation/copy code is in the Gallium r600 driver underneath it.

## Launch and model integration

- `tools/r600_auto_workgroups.c`: external OpenCL interposition shim, caps only
  unspecified local sizes at 32. Required/explicit kernel sizes remain intact.
- `tools/with-r600-clover.sh`: explicit opt-in profile combining the established
  flags, isolated driver selection and shim; no guessed reserve for 1 GiB cards.
- `examples/train_yolo_opencl.py`, `examples/fused_ema.py`: optional pinned
  Ultralytics adapter, CPU NMS/metrics, FP32 portable checkpoints and native EMA
  integration. Model architecture/YOLO loss math is not rewritten.
- `examples/ultralytics_memory.py`, `examples/training_telemetry.py`: meaningful
  allocator-peak display, epoch counters and bounded benchmark/epoch timings.

The Flask app and wildlife data/models remain outside this backend repository.
See the [profile/support guide](r600-clover.md) and [performance evidence](r600-performance.md)
for tested hardware and limits. A completed training run is not proof of every
public ATen API, dtype/layout, GPU or compiler configuration.
