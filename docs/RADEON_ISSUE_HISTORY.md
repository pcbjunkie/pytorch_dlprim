Latest release status: all five models passed the consolidated hardware gates.
The [release guide](RADEON_MINIMAL_BASELINE.md) is the current settings summary;
chronological entries below intentionally retain earlier pending states.

# Radeon OpenCL issue history and upstream review ledger

Updated 2026-11-03. This is the review index for **what changed, why, where,
and how we know**. Preserve upstream authorship/history. This ledger supplements,
rather than overwrites, the chronological [investigation notes](r600-history.md).
Old notes intentionally retain failed hypotheses and intermediate configurations.
[Layer ownership](changes-by-layer.md), [revalidation](r600-workaround-revalidation.md),
and [benchmarks](RADEON_BENCHMARKS.md) provide detailed supporting records.
Machine-readable issue entries: [radeon-issue-ledger.json](radeon-issue-ledger.json).

## Evidence rules

- Discovery hardware means the GPU on which a failure is documented, not every
  GPU later tested. Unknown first-discovery provenance is explicitly pending.
- Separate observed symptom, demonstrated defect, working workaround, and root
  cause. A passing workaround does not establish a compiler/hardware diagnosis.
- Standalone failures with every buffer alive cannot simply be attributed to
  premature Torch temporary destruction. Compiler selection errors happen before
  execution and are not explained by runtime buffer lifetime.
- Test a single change against the recorded candidate. Combined flags can mask
  each other; a pass with one flush does not prove both flushes removable.
- Short CPU parity, native throughput, sustained training and saved-model
  validation are distinct gates. Six epochs deliberately stopped is not20 epochs.
- Production launchers and frozen snapshots remain controls; experimental flags
  are not yet published defaults. No universal Radeon support claims.

## Issue register

| ID | Issue and ownership | Discovery / observed hardware | Latest disposition |
| --- | --- | --- | --- |
| MEM-01 | r600 overlapping compute-pool allocations; Mesa r600 compute_memory_pool.c | Exact first card not yet audited; do not infer from later training | retain placement fix |
| MEM-02 | GPU temporary lifetime and ordinary cache reuse; backend src/CLTensor.cpp | R9 270 ordinary-cache failure independently reproduced; earlier r600 investigation also involved temporary lifetimes | retain deferred retirement; underlying ordinary-cache failure not fully isolated |
| MEM-03 | initial pool reservation and tight allocations; Mesa pool plus backend allocator | per-card experiments; first discovery pending audit | retain tested per-card reservation; GCN uses ordinary allocation |
| SYNC-01 | copy-cache flush around pool migration; Mesa r600_hw_context.c | HD6970 Cayman | OFF in all three verified legacy profiles with per-kernel flush ON |
| SYNC-02 | per-kernel clFlush; OpenCL launch shim tools/r600_flush_after_kernel.c | HD5870 Cypress | retain on Barts candidate; removal caused standalone corruption or Torch SIGSEGV depending on copy-flush setting |
| TYPE-01 | packed byte stores and int8 scatter widened to FP32; DLPrim pointwise/util kernels and backend scatter_ops.cpp | HD6970 Cayman | OFF on Cypress/Barts; required ON for Cayman; all three completed 20 added epochs |
| WG-01 | automatic workgroup cap32 and X-first layout; launch shim tools/r600_auto_workgroups.c | Cayman lane-dependent indexing/copy failures; first introduction provenance still to audit | caps through256 pass isolated Cypress/Barts checks; Barts128 fastest tested; explicit local sizes unaffected |
| WG-02 | BatchNorm cap64 and matching workspace; DLPrim src/core/bn.cpp | HD6970 Cayman | 128 verified on Cypress/Barts; Cayman retains64 after larger-shape failures |
| GEMM-01 | small GEMM tiles; DLPrim src/gemm.cpp | historical M256 N1152 K128 hang documented; exact first card pending source-log audit | normal tiles verified on Barts; Cypress and Cayman require small tiles after controlled hangs |
| ALG-01 | disable Winograd; DLPrim src/core/conv.cpp and Winograd kernels | original first-card record pending audit; independently confirmed HD6850 Barts now | retain for training; LLVM AtomicCmpSwap selection abort in winconv_3x3_bwd_filter; forward-only not separately qualified |
| ALG-02 | disable ordinary GEMM split-K; DLPrim src/gemm.cpp and sgemm kernel | original first-card record pending audit; independently confirmed HD6850 Barts now | retain; LLVM AtomicCmpSwap selection abort in sgemm |
| ALG-03 | atomic-free convolution/pool gather and fixed launch geometry; DLPrim core/conv.cpp and core/pool.cpp plus kernels | convolution geometry hangs on HD6850 and HD6970; exact first atomic-compile failure provenance pending | retain; Barts removal controls failed LLVM AtomicCmpSwap selection |
| IDX-01 | 32-bit metadata, branchless divmod, cap32 and flattened strided copies; backend index/training_ops and DLPrim core/util.cpp | HD6970 Cayman standalone reproducers | 32-bit arithmetic fixes retained; index128 and original copy layout verified on all three legacy cards |
| FIX-01 | homogeneous pointwise destination storage offsets; DLPrim src/core/pointwise.cpp | HD6970 investigation; defect is not asserted hardware-specific | retain unconditionally, not a removable compatibility flag |
| FIX-02 | parallel top-k shared dynamic local storage; backend src/training_ops.cpp | R9 270 Pitcairn | retain V9 one-allocation layout; exact Mesa/compiler source cause not isolated |
| FIX-03 | scalar promotion, repeat/view fallback, activation gradients; backend tensor/pointwise and related modules | R9 270 early Clover/Rusticl checks; per-defect provenance needs audit | retain, do not revert with compatibility flags |
| PERF-01 | batched norms/SGD/EMA, fused SiLU and parallel top-k; backend foreach_norm/batch_pointwise/ema/pointwise/training_ops | R9 270 early tuning and HD5870 parallel top-k development | retain tested fast paths and fallbacks; publish native timing separately from profiles |
| PERF-02 | filter-gradient partials, gather vectorization and depthwise scheduling; DLPrim core/conv.cpp and kernels | legacy development; HD5870 backward tuning documented | retain validated parts; custom FILTER_SPLIT_K is separate from disabled ordinary split-K |

## Investigation sequence and decisions

1. Early R9 270 work established the custom-device bridge and implemented
   missing supported GPU operations; scalar, gradient and view correctness
   changes are distinct from later legacy-driver compatibility settings.
2. Barts/Cayman exploration exposed pool placement, copies/indexing, subword
   stores, launch geometry and temporary-lifetime failures. Preserve the native
   reproducers and failed controls; never summarize this as one known bug.
3. Standalone pointwise offset reproduction established an actual DLPrimitives
   output-offset defect. Branchless division/flat-copy fixtures established
   additional failures whose compiler/hardware root causes remain uncertain.
4. Deferred buffer retirement, pool preallocation, launch restrictions and flush
   settings produced working training profiles. Subsequent tuning and five-card
   phase-one baselines are retained with their exact backend/configuration.
5. R9 270 exposed parallel top-k local-memory corruption. V9 changes two dynamic
   local arguments into one allocation partitioned internally. Independent
   ordinary-cache failures still required the protected retirement profile.
6. On November1, V9 legacy revalidation passed on Cypress then Barts. Packed
   byte/int8 widening removal and larger automatic/BN groups passed focused
   checks and full-model comparisons. Barts auto128/BN128/packing-off gained
   about6.2% over its matched conservative control. This does not prove lifetime
   fixes caused the earlier subword/workgroup defects.
7. Barts candidate ran six additional epochs41-46 at8.5846 training images/s.
   The user intentionally stopped it; last checkpoint loaded on CPU with finite
   EMA weights.49 TTM placement warnings, no captured GPU reset/stall. Longer
   per-card validation is deferred until the common candidate settles.
8. Copy-cache flush OFF passed focused and model checks while per-kernel flush
   stayed ON. With BOTH off, native pool-growth copy failed trial0 with5120
   mismatches. With per-kernel OFF/copy-cache ON, native copies passed but the
   Torch copy process segfaulted. Restoring per-kernel ON/copy-cache OFF passed
   all99 copy checks. **SIGSEGV cause is unisolated**; distinguish a process
   crash from a GPU reset, and audit interposition/symbol resolution as well as
   runtime behavior before assigning driver blame.
9. Small GEMM tiles OFF passed original/fringe matrix and convolution checks,
   two model CPU updates and matched timing.8.52584 vs8.54277 images/s is
   effectively equal; removal is supported for the evolving Barts candidate.
10. Winograd and ordinary split-K were independently re-enabled. LLVM aborted
    selecting AtomicCmpSwap in `winconv_3x3_bwd_filter` and `sgemm` respectively.
    Retain those training restrictions. Custom filter-gradient splitting is
    unrelated and remains enabled. Control parity and native timing passed.

## Current candidate, not a universal default

Barts: V9 backend, automatic cap128, BN cap128, packed-byte outputs OFF, small
GEMM tiles OFF, copy-cache flush OFF, per-kernel clFlush ON, pool reservation960
MiB, no ordinary allocator cache, tight allocations, bounded deferred retirement
(queue128, retirement64 MiB). Keep indexing cap32/flat-copy, gather gradients,
Winograd disabled, ordinary split-K disabled, and existing validated performance
features. Not every kernel uses128 threads: explicit geometry is unchanged.
No full long run has tested this entire progressively simplified combination.

## Proposed upstream boundaries

| Destination | Scope | Evidence expected with the PR |
| --- | --- | --- |
| pytorch_dlprim | Dispatcher/operator correctness and supported GPU operators | Minimal CPU parity/gradient/stride/out/error reproducers and supported-case limits |
| pytorch_dlprim | Deferred lifetime/accounting and synchronization | Queued allocation churn, aliases, explicit drains, concurrency assumptions and memory counters |
| pytorch_dlprim | Batching/fusion/top-k performance | Correctness gates, native matched timing, fallback behavior; separate local-memory fix rationale |
| DLPrimitives | General output-offset/coordinate defects | Small upstream-base failure and passing fixed test, independent of model |
| DLPrimitives | Atomic-free algorithms and legacy tuning | Opt-in selection, numeric limits, geometry/workspace tests, per-card evidence |
| Mesa r600 | Pool placement and optional migration/cache changes | Standalone OpenCL/Gallium reproducer, exact Mesa base, minimal patch; preserve unknown root causes |
| Integration/docs | Per-card launch profiles, benchmarks and build reproducibility | Tested compiler/runtime/device versions; no application datasets or weights |

Do not submit the full experimental profile as one unexplained fix. First
review/split changes by ownership and causality; maintain optional tuning and
legacy compatibility independently from general correctness fixes. PyTorch
core, LLVM and the Linux kernel have not been patched by this work.

## Reproducibility and remaining upstream-preparation work

- Dependency bases/fork pins: `patches/r600-bases.json`. Runtime/source layers:
  `patches/dlprimitives-r600-clover.patch` and
  `patches/mesa-25.0.7-r600-clover.patch`. Regenerate and verify patches against
  current selected source before submission; historical binaries are not proof
  that the current patch reconstructs every newer experiment.
- Frozen V8 SHA256:
  `9a58c39375ac4e765643a9aac37267e986d6861d363cede990f5a4342f691394`.
  V9 SHA256:
  `5b25148c3f1d093a9f8b21ec8207e9fb5910cb9cf5e1f7cd02515310a8a46ed9`.
  Preserve per-run config, source/model hashes, raw logs and command lines.
- Root-relative `runs/...` references are local artifacts, not shipped repository
  content. Promote essential standalone fixtures/controllers into portable tests
  and prepare a focused evidence bundle before a PR. Do not make reviewers
  depend on this machine's absolute paths or unpublished binary snapshots.
- First-discovery hardware for small GEMM, original algorithm restrictions,
  pool placement and several early bridge fixes still needs log/source audit.
  Do not silently fill these gaps with the current card's name.
- Remaining public API limits include native BN saved-statistic semantics,
  NLL total_weight metadata, broader dtype/layout/view/out bounds, CPU fallback
  coverage, and unestablished multi-GPU parallel execution. Existing training
  success does not close these issues.
- Finish the focused Barts baseline reassessment, then targeted other-card
  revalidation with extra attention to original failure hardware. Collect each
  card's best verified settings; longer runs follow on the settled candidate.
- User authorizes a new baseline branch/push only after all cards pass. No new
  branch, commit, push or PR is created by this documentation update.

## How to keep this record current

For every new result, update its stable issue ID with discovery hardware,
failed/control/fixed configurations, exact reproducer, source layer/flag,
root-cause certainty, artifact path and validation limits. Record rejected
changes and intentional stops. Add a new ID for a different defect instead of
rewriting the old history to fit a later hypothesis.

## Subsequent focused Barts decisions

Index cap128 and original rank3 copy layout together passed focused/modelCPU
checks and nativebenchmark8.626i/s vs8.544control (~0.97%). Retain arithmetic
fixes; other-card confirmation remains pending. Removing convolution and
pooling gather separately then failedLLVM AtomicCmpSwap selection in sgemm
andpooling_bw. Controlpassed. These join Winogradbackward-filter and ordinary
split-K as four distinct paths blocked by the same observed compiler primitive.
Keep their individual reproducers; do not infer buffer-lifetime causality.
See `runs/hd6850-index-copy-benchmark-20261101/` and
`runs/hd6850-gather-revisit-20261101/`.

## Consolidation packaging note: launch-shim linkage

The new configurable shim uses the repository build script, explicitly linking
libOpenCL; historical one-off auto128 experiment shims linked only libdl/libc.
The prior no-perkernel Torch SIGSEGV remains unisolated and may involve
interposition/dlsym symbol resolution. Do not cite it alone as proof of a GPU
ordering defect. The independent standalone BOTH-flushes-OFF copy mismatch
still stands. Minimal-v1 conservatively retains the previously working
per-kernel flush; removing it with a correctly linked shim is a separate future
audit, not a new consolidated-baseline claim.

## Minimal-v1 consolidation checkpoint

Legacy andGCN launch paths are separate. Newconfigurable workgroupshim uses
repo buildscript, default32 and explicit32/64/128/256caps; compiled into a new
path, old binaries preserved. Legacyhelper adds explicitminimal-v1 while old
conservative behavior remains default. Byte/BN/GEMMfixtures now allow the
native/newcaps and historicalM256/N1152/K128 fixture is portable in tests/local.
Bartsconsolidation passes11suites,CPU2updates,inference800 with2TTMwarnings
andnoGPUfault. Decoded13.235FPS trails historical15.552FPS; this unmatched
configuration observation is a performance followup, not a confirmedcause or
permission to discard earlier inferenceprofile. No speedclaim fornewlinkage.
NextHD6970 (userselectedoriginalfailurehardware),then5870/R9270/7970 short
confirmation,thenlongrunEACH,thennewbranch/push. No pushperformed.

## Cayman confirmation: two historical restrictions remain

Onreturning6970, commoncandidate nativebytes failed1element comparison;
restoringpackingalonepasses772checks. SubsequentBN128 test passes8module
cases then hangslarge2stage reduction, with persistentkernelGPUlockup/ring0
stalls. Testcontroller stopped; no moreGPUwork before reboot. Packing1/BN64
is proposednextcontrolledCaymancandidate, not yet qualified. PreserveCypress/
Barts removals; do not generalize either result acrossVLIW4andVLIW5.

## LLVM Cayman random-fill defect isolated; scope decision

Uniform RNG standaloneLLVM15 compilation emits a130-slot ALU clause with
count129 encodedas1 (seven-bitfield). CPU-generated.text matches cachedhung
kernelbyte-for-byte. Nativeexplicit32 alsohangs; therefore126orlargegroups
arenotnecessaryforthisfailure. OfflineLLVMsource investigation suggests
vector-literal undercounting; candidatepatchnotvalidated. Userdirecteddeferring
compilerfix/rebuild toLLVM,keepingtopkregressioninputCPUside. Testchangedonly
GPUrandtoCPUrand.to(ocl); no productionRNGfix/fallback, sourceLLVM unchanged.
CPU-onlyupstreamreproducer/report: docs/LLVM_CAYMAN_RNG_BUG.md and
reproducers/llvm15-cayman-rng/. Reportnot submitted. GPU RNG remainsknown
unsafeunderthisconfiguration; modelsusingitrequireseparatequalification.
No inference thatBN/GEMMhangssharetheclause-overflowcause.

## Cayman confirmation after CPU test-input correction

Topk129checks nowpasswithCPU-generateddata. Nativeautomaticcap128 then
producedNaNlossaroundbatch29 andscatterbounds later, noGPUErrors. Changing
ONLYauto32 yields two76stepnativepasses14.2455/14.2001i/s,CPU2updateswith
cosines.99999980/.99999984 and800inferencecomparisons. No newkernelmessages.
Adoptcayman-minimal-v1 candidate(auto32/BN64/packed1/smalltiles1) whilekeeping
index128/originalcopy andcopycacheflushOFF. Retainperkernelflush/lifetimefixes.
This numericfailure has no established linktotheRNGclauseoverflow. No final
sustainedqualification; remainingother-cardconfirmations precede longruns.

## Cypress confirmation: smalltiles retained, other removals pass

NormalGEMMtiles hangsordinaryM256/N257/K128 aftersmaller/conv fixturespass.
Afterreboot ONLYsmalltiles1restored, thenallremainingchecks/modelCPU2/native2/
inference800pass. Native12.987/13.248i/s; decoded17.404FPS; no kernelmessages.
Namedcypress-minimal-v1 retains auto/BN/index128,nativebytes,originalcopy,
copycacheflushOFF. Bartsdoesnotneedsmalltileoverride; Caymankeeps additional
packing/BN64/auto32protections. Runtime/compilerdefectcauses remainseparate;
GPU RNG compilerreport is not asserted toexplain theseGEMMfailures.


### 2026-11-02: Cypress consolidated baseline sustained pass

The cypress-minimal-v1 profile completed epochs 41–60 at batch 24, reserve 960 MiB: 13.0244 training images/s overall, 10637.82 seconds including validation/checkpointing. Exit 0; best checkpoint reloaded and validated. User live dmesg capture remained active with zero new bytes. No profiling, per-batch debug hooks or CPU spot checks. Evidence: runs/hd5870-minimal-v1-stability20-20261102. Current GCN confirmations remain pending; no new branch or push yet.


### R9 270 consolidated baseline sustained pass, 2026-11-02

Epochs 41–60 completed, batch 48, ordinary allocation/no r600 pool. Training 14.9820 images/s overall, 2.40 hours including validation/saves. Exit 0, best checkpoint reloaded and validated; active dmesg capture recorded zero new bytes. No profiling or CPU spot checks. Final best-checkpoint validation mAP50 0.698, mAP50–95 0.421. Evidence: runs/r9-270-minimal-v1-stability20-20261102/stability-summary.json. HD7970 current-candidate confirmation next.


### HD7970 consolidated baseline sustained pass, 2026-11-02

Epochs 41–60 completed at batch 80 with ordinary GCN allocation and protected deferred retirement. Training 20.6131 images/s overall; 1.76 hours including validation/checkpointing. Exit 0, last.pt reloaded and validated. Active live kernel capture remained empty. No operator profiling or CPU spot checks. Evidence: runs/hd7970-minimal-v1-stability20-20261102/stability-summary.json. Short inference decoded14.5596FPS with independent timing-window inconsistency; do not interpret as resolved inference optimization. Barts/HD6850 and Cayman/HD6970 final sustained runs still pending before release branch/push.


### ASUS HD6850 final sustained pass

20 added epochs41–60 passed, batch24/reserve960; 8.8264 training images/s overall, 4.27 hours total. Saved best checkpoint reloaded and validated, no new captured kernel bytes (initial54-byte network message unchanged). Single HD6850 entry retained, ASUS1043:03b4 board evidence separate. Evidence runs/hd6850-asus-minimal-v1-stability20-20261102/stability-summary.json. Only HD6970 final current-candidate sustained test remains before branch/push.


### Cayman final sustained pass, 2026-11-03

Epochs41–60 completed at batch28/reserve1008 with cayman-minimal-v1, no profiling/CPU spot checks. Training 14.2966 images/s overall; 2.74 hours including validation/saves. Exit0, best checkpoint reloaded and validated. Active kernel capture remained empty. Evidence runs/hd6970-minimal-v1-stability20-20261102/stability-summary.json. All five GPU models now have current-candidate short and20-epoch sustained passes; source/document review and new branch/push remain. HD5870 selected permanent deployment/reference card; user owns three.


### Release integration review, 2026-11-03

All five models completed the consolidated short checks and 20 added epochs.
Regenerated combined DLPrimitives patch from the pinned base, including all new
kernels; applying it to a pristine tree reproduced all 17 changed/new files
exactly. The gitlink remains at upstream base intentionally. Mesa is unchanged;
its two-file compatibility patch matches the existing pinned fork.

Merged current fork master, preserving upstream's empty-tensor torch.cat fix.
The integrated backend built and imported under Torch2.4.1. Three C shims built;
Python/JSON/shell syntax and all three legacy profile/GCN isolation checks passed.
This host review does not replace the frozenV9 five-card hardware qualification;
the upstream cat integration has not been rerun on those GPUs. No installed
Torch core, driver, dataset, model, or Squirltool changes are part of the release.
See benchmarks/release-validation-v1.json for review evidence.
