# Verified Radeon OpenCL baseline v1

Updated 2026-11-03. All five GPU models passed focused correctness/model checks
and 20 additional wildlife-training epochs with the consolidated source.
These are explicit opt-in profiles, not a claim that arbitrary OpenCL models,
dtypes, drivers, or Radeon cards work. The old `conservative` profile remains
available. See [issue history](RADEON_ISSUE_HISTORY.md) for retained failures and
[benchmark records](benchmarks/radeon-baseline-release-v1.json) for exact results.

## Verified results

YOLOv8n, five wildlife classes, stock Ultralytics 8.3.0, PyTorch 2.4.1,
256×256 FP32; Phenom II X4 955, 16 GiB RAM, Debian 12, Linux 6.1.0-42,
headless, two Torch threads and one training loader worker.

| GPU | Profile | Batch | Training reserve MiB | Training images/s | Detection FPS | Stability |
| --- | --- | ---: | --- | ---: | ---: | --- |
| HD 5870 | cypress-minimal-v1 | 24 | 960 | 13.02 | 17.40 | 20 added epochs passed |
| HD 6850 (ASUS) | minimal-v1 | 24 | 960 | 8.83 | 14.73 | 20 added epochs passed |
| HD 6970 | cayman-minimal-v1 | 28 | 1008 | 14.30 | 15.28 | 20 added epochs passed |
| R9 270 | GCN | 48 | ordinary buffers | 14.98 | 10.99 | 20 added epochs passed |
| HD 7970 | GCN | 80 | ordinary buffers | 20.61 | 14.56 | 20 added epochs passed |

Training throughput is total training images divided by measured training time;
it includes loading, augmentation, forward/loss/backward, optimizer/EMA and batch
logging, and excludes validation/checkpoint writes. All runs completed epochs
41–60, reloaded and validated a saved checkpoint, and recorded zero new kernel
log bytes during the run. This is observed test coverage, not a universal driver
stability guarantee. There was no operator profiling or periodic CPU comparison
in these sustained runs; separate short parity tests preceded them.

Detection is batch 1, fused model, decoded image through GPU forward and CPU NMS,
with preprocessing, two Torch threads and one OpenCV thread. It uses a fixed
checkpoint and SHA-verified 20-image manifest: 20 warmups and 100 timed frames per
mode, plus 800 CPU/raw-output/NMS comparisons outside timing. Legacy inference
uses a smaller 128 MiB reservation and queue limit 64. Independent inference
windows are inconsistent on GCN (sometimes encoded or decoded pipelines measure
faster than forward-only); preserve those results but repeat/profile before
claiming an inference regression or optimum. Training batches are verified
working choices, not new exhaustive maximum-capacity sweeps.

Both tested HD6850 boards share one model entry: the ASUS EAH6850 short training
result was within 1.8% of the other board. The final full run used the ASUS.
The HD7970 result retains the earlier nonstandard auxiliary-power qualification;
no measured board-power claim is made. See its per-card report.

## Legacy profiles: select by GPU

| GPU | R600_PROFILE | Auto cap | BatchNorm cap | Index cap | Packed bytes | Small GEMM tiles |
| --- | --- | ---: | ---: | ---: | --- | --- |
| HD5870 / Cypress | `cypress-minimal-v1` | 128 | 128 | 128 | off | on |
| HD6850 / Barts | `minimal-v1` | 128 | 128 | 128 | off | off |
| HD6970 / Cayman | `cayman-minimal-v1` | 32 | 64 | 128 | on | on |

All three retain per-kernel `clFlush`, the patched Mesa compute pool and tested
reservation, tight allocation and protected deferred buffer retirement. Queue
limit 128 and retirement budget 64 MiB were used for training. Gather convolution
and pooling gradients, disabled Winograd/ordinary GEMM split-K, and tested custom
filter-gradient splitting/fusions remain enabled as selected by the launcher.
Original rank-3 copy layout is used; extra Mesa copy-cache flushing is unset.
The copy-flush flag checks presence: setting it to `0` does not disable it.

Cayman removal controls failed for native byte stores, larger BatchNorm/automatic
workgroups, and normal GEMM tiles. Cypress also hung with normal GEMM tiles.
These exceptions are retained; don't substitute the Barts profile on those cards.
An automatic cap is not an exact local width (it chooses divisors of global
sizes); explicit or compiler-required workgroup geometry is not overridden.

Apply the [combined DLPrimitives patch](../patches/dlprimitives-r600-clover.patch)
to the pinned clean upstream submodule as described in the [build guide](r600-clover.md).
The gitlink intentionally stays at that upstream base; the patch includes both
tracked edits and new kernels. Do not apply the portable-reductions patch again.
Legacy profiles also require the pinned [Mesa fork](https://github.com/pcbjunkie/mesa/tree/r600-clover-25.0);
the general backend and GCN profiles do not require that r600 runtime.

Build the shims before launching (paths are relative to this repository):

```sh
R600_SHIM="$PWD/build-r600-minimal-v1/r600_auto_workgroups.so" tools/build-r600-shim.sh
tools/build-r600-flush-shim.sh
```

Select your isolated ICD, patched Mesa prefix/pipe and built backend according to
the build guide, then for example on Cypress:

```sh
R600_PROFILE=cypress-minimal-v1 R600_FLUSH_AFTER_KERNEL=1 \
  R600_SHIM="$PWD/build-r600-minimal-v1/r600_auto_workgroups.so" \
  R600_POOL_RESERVE_MB=960 OPENCL_ASYNC_OP_LIMIT=128 \
  OPENCL_DEFERRED_RELEASE_MB=64 tools/with-r600-clover.sh \
  python tests/local/check_flat_strided_copy.py
```

## GCN profiles: R9 270 and HD7970

Use `tools/with-radeon-gcn-baseline.sh` with your system Clover ICD and selected
backend (`DLPRIM_PYTHON_PATH`, default `build`). It selects ordinary allocations,
tight/no-cache allocation and protected deferred retirement, queue limit 128,
retirement budget 64 MiB. It clears the legacy overrides and launch shims. No
r600 pool reservation or per-kernel flush shim is used. A common source baseline
does not imply identical runtime settings on r600 and GCN.

## Known limits and release provenance

The frozen tested extension is V9, SHA-256
`5b25148c3f1d093a9f8b21ec8207e9fb5910cb9cf5e1f7cd02515310a8a46ed9`.
Frozen binaries and raw local runs are not distributed in this source repository.
The release evidence records their paths/hashes for audit. The source merge also
preserves upstream's empty-tensor `torch.cat` fix; this integration is separately
build-checked and has not been rerun through the five-card hardware suite.

Cayman/LLVM15 GPU random fill is known unsafe: the compiler emitted an oversized
ALU clause even with explicit local width32. Regression inputs were generated on
CPU and transferred to the GPU. This is not a fixed GPU RNG implementation.
See the [LLVM report and CPU-only reproducer](LLVM_CAYMAN_RNG_BUG.md). No newer
LLVM compiler verification is claimed. Shapes/dtypes outside the exercised paths
may still fall back to CPU or fail; multi-GPU operation is unqualified.

The optional K4200 CUDA comparison produced no benchmark: direct driver
initialization returned802 on the MCP61 motherboard; the older-driver probe
was stopped at the user's debugging cutoff. It is not evidence about CUDA
performance. Squirltool, datasets, weights and CUDA workspaces are separate.

The original per-change investigations remain in
[r600-workaround-revalidation.md](r600-workaround-revalidation.md), and earlier
phase-one results remain in [RADEON_BENCHMARKS.md](RADEON_BENCHMARKS.md).
