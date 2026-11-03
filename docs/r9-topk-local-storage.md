# R9 270 top-k local-storage regression

The V8 parallel top-k kernel passed the recorded r600 tests but failed on
PITCAIRN with system Mesa25.0.7 Clover/radeonsi. The first batch2 full-model
check returned loss17.97966 against CPU9.93948. Raw network outputs agreed
within9e-6; isolated network layers also passed. Disabling Winograd and,
separately, buffer caching did not remove the V8 loss mismatch.

A differential loss trace isolated `aten.topk.default`. The saved input has
shape2x1x1344 and k10. Output scores contained index bits interpreted as FP32:
index1316 appeared as1.84411e-42, followed by repeated63/8.8282e-44. A native
OpenCL harness reproduced this without Torch or YOLO.

The original kernel passes two dynamic local arguments, one FP32 score array
and one uint32 index array. Replacing them with a single dynamic local argument
and explicitly partitioning its storage into the two regions made the native
reproducer pass ten repeated dispatches. This supports a local-argument layout
aliasing diagnosis; the exact responsible Mesa/compiler source is not isolated.

Only `src/training_ops.cpp` local-storage declarations and host argument binding
changed. The selection algorithm, comparison order and workgroup width remain
the same. No driver, YOLO or global allocator changes were made by this fix.

The frozen V9 binary SHA256 is
`5b25148c3f1d093a9f8b21ec8207e9fb5910cb9cf5e1f7cd02515310a8a46ed9`.
`tests/local/check_topk_local_storage.py` passes54 repeated comparisons;
`tests/local/check_topk_parallel.py` passes129 cases, including views, ties,
nonfinite values, integer/non-last-axis fallbacks and autograd. Its preexisting
`aten::scatter.src_out` autograd fallback warning is retained, not suppressed.

The full-model batch2 gate then passed both CPU loss/gradient comparisons.
Batch24 still fails normal execution (GPU132.6949 vs CPU89.3241); raw network
outputs agree within1.6e-5, and per-operator loss tracing on identical CPU
predictions produces the correct loss. Investigation of that additional
execution/lifetime issue is ongoing. V9 is not yet a completed R9 training
baseline. Legacy-card V9 regressions also remain pending; historical V8 card
results and binaries are preserved.

Workspace evidence includes `runs/r9-topk-local-original.log`,
`runs/r9-topk-local-combined.log`, `runs/r9-topk-v9-regressions.json`,
`runs/r9-b24-loss-operator.log` and the failed baseline trial directories.

The V9 synchronous no-cache control subsequently passed both batch24 full-model
CPU loss/gradient comparisons. The existing no-cache/tight-allocation/deferred
retirement profile (queue128,64MiB retirement budget) also passed both updates
at batches2 and24. At batch24 both scalar losses matched CPU exactly and
gradient cosine exceeded0.999999999994. This is a tested execution workaround;
the default-cache reuse/ordering defect has not been source-isolated or fixed.
The clean baseline sweep resumes using that explicitly recorded profile, with
no r600 shader shim or global-pool reservation. Long stability remains pending.
