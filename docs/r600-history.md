# Historical investigation notes

These notes record intermediate failures and provisional results. They are not
the current support status. See [r600 profile](r600-clover.md) and
[LOCAL_CHANGES](../LOCAL_CHANGES.md) for the selected configuration.

> Latest BN compatibility result: DLPRIM_BN_MAX_WG=128 passed eight full module checks, then hung in the first stage of a larger two-stage statistics reduction. It is not validated training support. The earlier DLPRIM_BN_SINGLE_WAVE flag was replaced by DLPRIM_BN_MAX_WG (64/128/256); default unchanged. Training wrapper stays disabled.

> Subsequent full YOLO probe crashed and required reboot before any optimizer step. The 8x8 change is only a passing small-case workaround, not a stable training fix. The experimental HD6970 training wrapper is disabled.

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

## Experimental legacy-r600 max-pool gather

DLPrimitives core pooling has an opt-in `DLPRIM_POOL_GATHER=1` maximum-pooling backward path. Each thread owns one input gradient and gathers contributions from output windows that selected it, avoiding floating-point CAS. The host dispatches over input spatial dimensions; existing gradient scaling/accumulation is retained. Average pooling and the default atomic path are unchanged. The gather specialization excludes unused atomic code from compilation.

Verified on HD6850 / patched Mesa25 Clover with the conservative profile:12 CPU-referenced finite-input cases cover ties, overlapping/nonoverlapping windows, gaps, rectangular kernels, padding, ceil mode, multiple batches/channels, and kernels3/5/9/13; a three-stage SPPF chain also matches CPU accumulated gradients. Test: tests/local/check_pool_gather.py. No HD5000/5900 hardware test yet. This does not establish full legacy-card training: the next real-data run fails in GPU unique() loss preprocessing with an invalid count.

## Legacy32 small-list unique

On32-bit-address OpenCL devices, supported small-list unique now uses uint32 rank/order/count metadata and parallel destination-owned compaction. It writes ATen's int64 counts/inverse outputs using endian-aware pairs of32-bit words; int64 input comparison uses signed high-word/unsigned low-word ordering, preserving the full int64 range. The existing64-bit-address implementation is retained. Supported dtypes remain FP32/int32/int64 with at most4096 elements; preexisting larger/unsupported input fallback is unchanged. Only the final scalar output size is read by the host. The legacy no-cache compatibility profile synchronizes the two stages before dependent access/readback.

On HD6850/patched Mesa25 Clover, tests/local/check_unique_legacy_regression.py passes336 CPU-reference combinations: all sorted/inverse/count flags, empty/scalar/strided/offset inputs,4096-element inputs, float NaNs/infinities, and full signed integer limits. No fresh R9 or HD59xx hardware validation.

Final tests also include4096 distinct values (maximum output cardinality) and GPU-created strided image-ID columns with storage offsets0/6/18, matching YOLO target preprocessing. Output staging tensors are retained until their copies complete in synchronous compatibility mode.

## Experimental legacy32 indexing and convolution compatibility

The 32-bit-address advanced-index read/nonzero path uses checked uint32 coordinate metadata, inline software divmod and endian-aware int64 output words. Bounds errors use per-output flags and a GPU reduction instead of atomics. In synchronous/no-cache mode temporary coordinate buffers survive dependent copies. Indexed writes use a physical-address map respecting offsets/strides; repeated-index accumulation uses destination-owned gathering to avoid CAS, with explicit integer wrapping. The 64-bit-address implementations remain in place. CPU comparisons on HD6850 covered reads/writes, masks, strides, broadcasting, errors and autograd. On a fresh HD6970 boot, the full loss-indexing/autograd suite and336 unique combinations also pass. Relevant tests are in tests/local.

An opt-in DLPrimitives `DLPRIM_CONV_GATHER=1` convolution input-gradient algorithm gathers into each destination pixel, avoiding floating-point atomics. Dense/grouped/depthwise shapes use FP32 with checked uint32 address bounds. The host MUST use the tested padded8x8x1 local workgroups: the original implementation-selected layout triggered ring0 hangs and GPU resets on both HD6850 and HD6970 (different PSU). Changing only launch geometry fixed the original Torch fixture on HD6970. The rebuilt native path passed8 CPU-referenced input-gradient cases and3 full convolution autograd cases (including weight/bias gradients) with no live kernel-log messages. This is a launch workaround, not a resolved driver/compiler root-cause diagnosis. Fresh HD6850 and HD59xx revalidation is still outstanding; no new throughput claim. See tests/local/check_legacy_conv_gather.py. The path remains explicitly opt-in and does not replace the normal algorithms by default.

## HD6970 conservative compute profile (experimental)

`DLPRIM_BN_MAX_WG=64` replaces the earlier single-wave flag and caps both reduction stages, with matching workspace. Ten BatchNorm module CPU comparisons pass, including two-stage reductions. WG128 passed smaller cases but hung a larger two-stage case; WG256 hung in a standalone captured-kernel reproducer. This does not establish a universal hardware thread limit. The separate native_batch_norm saved-variance/API convention remains unresolved.

`DLPRIM_GEMM_SMALL_TILES=1` selects32x32 output tiles,4x4 blocks,K32 and8x8 local workgroups for non-Intel GEMMs. Standalone filter-gradient reproduction and convolution/mm CPU fixtures pass. Default algorithms remain unchanged.

`DLPRIM_PACKED_BYTE_OUTPUTS=1` avoids failing shader subword stores: pointwise byte outputs stage into32-bit words, pack four bytes per word, then copy logical bytes using the OpenCL buffer API. Byte strided copies use buffer copies; noncontiguous type conversions materialize through the packed path. Boolean conversions compare against zero. Small int8 scatter counts compute exact integer values in FP32 then convert back through packed outputs. Shader byte/halfword stores were independently observed to zero the surrounding word; exact LLVM/driver cause is still unproven. This is an opt-in compatibility workaround, with synchronization overhead, not a performance optimization.

Legacy FP32 topk for k<=32 keeps selected indices in a private array and writes endian-aware int64 pairs, avoiding repeated indices observed in the prior64-bit loop. Legacy index_put accumulation now scans at most64 updates per dispatch, with explicit64-thread workgroups; destination ownership and update order are preserved. Other dtype/large-k paths retain their existing behavior.

HD6970 tests:772 pointwise/byte checks, int8 scatter cases, expanded mask/boolean conversion and unique-value topk cases pass. Chunked indexed accumulation passes repeat/boundary fixtures through1280 updates and a large sparse gradient layout. Positive-target YOLO criterion trace20261026-163610 completes868 kernels: all assignment outputs exactly CPU-match, loss607.401977539 CPU vs607.402038574 GPU, all three prediction-gradient max errors7.45e-9. No reset on that test. Full network training remains under test; these results are not a completed training claim. No fresh HD6850/HD59xx validation of this profile.

Two real HD6970 optimizer updates subsequently passed finite-loss/gradient and
weight-change checks (loss10.5487 then9.16986), no kernel faults. Full dataset
30-epoch burn-in is launched separately; completion/stability is not yet established.

## General homogeneous pointwise destination offset fix

The homogeneous pointwise generator loaded inputs with storage offsets but stored
outputs at index0, ignoring each output offset. A deterministic explicit scalar
multiply on a three-element nonzero-offset view overwrote preceding parent data.
Corrected stores to index+py_offset, including actual staging-buffer offsets in
packed-byte mode.417 CPU comparisons pass across FP32/int32/uint8/int8, view
offsets0/1/2/7/63/64/65/127, lengths1/3/64/65, multiply/add/pow, surrounding
storage preservation and shared three-scalar loss storage. Test in tests/local.
This fixes a reproduced bug; causality for the intermittent training failure is
still unproven. Default behavior is corrected for all supported devices.

## Separately tracked NLL metadata defect (not fixed)

Allocation poisoning exposes an unwritten total_weight scalar returned by
aten.nll_loss_forward. For reduction=none CPU returns zero, while the local
wrapper leaves allocator contents. YOLO's unreduced path does not consume it,
and this backend's backward ignores that argument. The per-sample losses in
that fixture are correct, so this is not established as the training cause.
Evidence: runs/hd6970-poc/nll-uninitialized-total-weight.pt. Keep pending for
API cleanup, including existing mean/ignored-label normalization semantics.

After the pointwise offset fix,32 retired-buffer stress iterations and3 full
forward/backward runs with fresh allocations filled by diagnostic NaN patterns
pass for the captured model/batch. The unused NLL metadata was excluded from
that finiteness probe after recording it separately. No claim that the
intermittent training corruption is resolved. CPU-reference training smoke
comparisons now include full gradients (cosine >.9999999999 in two updates).


## 2026-10-27: branchless legacy32 division helpers (under validation)

Both coord_divmod/nonzero and index_divmod/advanced indexing now use the
algebraically equivalent branchless restoring-division update. The captured
[2,1344] foreground mask exposed a wrong coordinate that caused an all-zero
indexed row. Standalone OpenCL with CPU-provided prefixes reproduces the old
helper failure at trial27; equivalent branchless version passes2000, including
on the original isolated pool-fixed driver. Scalar instead of vector stores,
additional copy flush flags, and promotion waits did not eliminate the old
failure. Exact compiler/hardware fault is not yet proven.

Frozen backend profiling-backends/diagnostic-branchless-20261027-055103.
Captured Torch nonzero plus boolean-index200 repetitions pass with experimental
copy-cache-flush driver and X-first automatic workgroup cap64. Strict full-loss
operator comparisons in progress. See runs/hd6970-poc/NONZERO_BRANCHLESS.md.
No full-training claim; normal profile and system Mesa remain unchanged.


## 2026-10-27: experimental flat rank3 strided copies

DLPRIM_FLAT_STRIDED_COPY=1 selects a 1D flattened dispatch for rank3 copies
on checked32-bit-address layouts, with shape constants for compiler division
and explicit local<=64. Tensor values stay on GPU. Default remains unchanged.
Standalone 3D strided copy reproduces one wrong value; a second GPU verifier
confirms it independently of host readback. Float-only/uint-only explanations
were rejected after a longer integer test also failed. A speculative bitwise
copy change was reverted. Flattened uint250 repetitions pass. Flattened float
stress reached953 iterations without mismatch during180s (timeout124, NOT a
completed2000-iteration pass). GPU and CPU both verified each completed trial.

Frozen candidate profiling-backends/diagnostic-flatcopy-20261027-060551 passes
99 copy comparisons (FP32/int32/int64, offsets, strides, tails, broadcasts and
unchanged destination neighbors). Strict loss replay and training validation
remain pending. Profile with-hd6970-diagnostic-fixes.sh selects this frozen
backend, the isolated driver copy-cache-flush candidate and auto-workgroup cap.
See runs/hd6970-poc/COPY_POOL_ORDERING.md for failed controls and limitations.


## 2026-10-27: WG32 candidate clears strict loss replay

Later flat64 tests still failed on five-channel copies. All isolated failing
work-item IDs examined were37 modulo64. Original coordinates atlocal32 pass200;
flat5 destination-order copy atlocal32 passes250 with GPU/CPU verification.
This is lane-dependent evidence, not a proven compiler or hardware attribution.
Experimental rank3 flat copies now use local<=32. DLPRIM_INDEX_WG optionally
caps nonzero prefix-scan width; diagnostic profile selects32. Padding is checked
against uint32 limits and shape-product constants use unsigned arithmetic.

Frozen profiling-backends/diagnostic-flat32-20261027-061506 passes99 copy
comparisons, existing indexing/bounds/out/gradient suite, and64 strict
operator-by-operator criterion replays (16 captured batches repeated4 times).
The diagnostic checker reads contiguous backing storage directly and constructs
views on CPU so it does not verify GPU strided copy using that same operation.
A100-update real training probe is running with CPU loss/full-gradient checks
every10 updates; no long-run reliability claim yet. Reproducers under
 tests/local/r600 and self-contained Python regression tests are retained.


WG32 real-training follow-up:100 optimizer updates completed with11 CPU loss/
full-gradient comparisons. Minimum cosine0.9999999451. Authorized guarded30epoch
run started at runs/hd6970-diagnostic-20261027-062754, CPU comparison every50.
Exact lane-dependent root cause and full burn-in remain unresolved. Summary:
runs/hd6970-poc/flat32-validation-summary.json. No changes to original app/model.
