# Radeon OpenCL backend development snapshot

This branch preserves upstream pytorch_dlprim history. It is an experimental development snapshot, not a general hardware-support or performance claim. Wildlife application, training adapters, datasets and model checkpoints belong in the separate squirltool project.

## Tested environment

Debian 12, Linux 6.1, AMD Phenom II X4 955, Radeon R9 270 (Pitcairn, 2 GB), Mesa 25.0.7 Clover, Python 3.11, PyTorch 2.4.1 CPU wheel. A separate Mesa 23.1.9 Rusticl runtime was also used for earlier checks. The latest operator pass was checked on Clover. Older VLIW Radeon cards have not yet been tested.

## Changes

- Device discovery/selection/properties and current-device allocation behavior.
- Mixed scalar promotion, hardtanh/ReLU6 gradients, and repeat/view-fallback correctness fixes.
- GPU vector norms, masking/where and small-axis scatter operations.
- Batched L2 norms/scalar packing; grouped foreach updates for native SGD/clipping; fused grouped EMA.
- GPU nonzero and advanced-index reads; indexed writes with repeated-index FP32 gradient accumulation.
- GPU addmv, small-k topk, small-list unique/counts/inverse, and FP32 max pooling with spatial indices.
- Per-device allocator live/reserved/cache/peak counters exposed through torch.ocl. Counters exclude driver allocations and other processes.

The full wildlife run completed 100 epochs using an earlier frozen backend. The subsequent operator pass passed CPU-reference tests, two CPU-matched YOLO steps, and a short complete training/validation/checkpoint-reload run with no warnings. No throughput gain is claimed for that pass.

## Build and runtime patch

Follow README-build.md. Keep the upstream dlprimitives submodule at its recorded commit. An optional separate runtime patch is stored in patches/dlprimitives-portable-reductions.patch; apply it inside the submodule with:

```sh
git -C dlprimitives apply ../patches/dlprimitives-portable-reductions.patch
```

This adds DLPRIM_CUSTOM_REDUCE=1 to select portable reductions for the tested older Rusticl runtime. The patch is retained here for reproducibility and can be proposed independently to DLPrimitives. The submodule gitlink remains upstream; no unpublished commit is required to clone this branch.

## Checks

After installing the backend, run the scripts in tests/local with an interpreter that imports the newly built pytorch_ocl. They require torch and numpy; see script imports for other optional requirements. They check values, gradients, alias/version behavior and empty/error/stride cases against CPU. They are not inference or end-to-end performance benchmarks.

## Limits before upstream submission

Some operators only accelerate the FP32/dtype/shape cases used by these models. Unsupported cases keep CPU paths and diagnostics. Unique rank sorting is capped at 4096 elements; topk uses repeated selection suited to small k; non-FP32 index accumulation uses simple destination-owned reduction. Broader optimization is future work.

Resolve the existing native_batch_norm saved-variance versus ATen saved-inverse-standard-deviation convention before claiming complete ATen compatibility. Audit general view/alias/out behavior and arbitrary strided-storage allocation bounds. Multi-GPU enumeration exists; DataParallel/DDP OpenCL execution is not established. Split this snapshot into focused, reviewed PRs before upstream submission.
