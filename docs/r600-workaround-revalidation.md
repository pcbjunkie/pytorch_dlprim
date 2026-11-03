# r600 workaround revalidation — 2026-11-01

Purpose: test whether current lifetime/retirement handling allows removing
historical correctness workarounds. Preserve frozen V9 and conservative default
launchers. Test one change at a time; a Cypress pass does not qualify Cayman.

| Workaround | Original observation | Revalidation |
| --- | --- | --- |
| Packed byte stores and FP32 int8 scatter | Neighboring byte corruption on Cayman | Cypress: 772 byte checks and 12 scatter cases pass both enabled and disabled; two batch24 model CPU loss/gradient updates pass disabled. No proven speed gain. Cayman pending. |
| Automatic local cap32, X-first | Lane-dependent copy/index corruption | Cypress copy/packed-byte checks passed at automatic caps32/64/128/256; explicit local shapes unaffected by this shim. |
| BatchNorm cap64 | Larger two-stage reductions hung on Cayman | All ten module forward/stats/backward/eval cases passed at each cap64/128/256 on Cypress. |
| Index/nonzero cap32 and flat rank3 copies | Wrong coordinates/copies, lane37 modulo64 | Pending separate cap32/64/128/256 checks and original captured reproducers; changing automatic cap alone does not test explicit index cap. |
| Small GEMM tiles | Filter-gradient dispatch hung with default algorithm | Pending default versus small tiles, original captured convolution fixture plus padded cases. Algorithm/tile change, not just thread count. |
| Gather input-gradient8x8 (spatial4 uses4x16) | Driver-selected layout hung on both Barts and Cayman | Pending explicit launch-shape candidates; preserve current gather arithmetic and padded global coverage. Requires candidate build or exact native reproducer. |
| Chunked index accumulation64 updates/dispatch | Legacy indexing correctness | Pending chunk-size experiment with repeated-index/boundary tests, distinct from workgroup width. |
| Per-kernel flush and copy-cache flush | Wrong predictions/copies | Keep enabled during above tests; separate experiment later, do not mix with workgroup changes. |

Cypress cast benchmark, V9 batch24, reservation960 MiB, one worker,
six warmups/24 measured: packed control A11.8997, native11.4453,
packed control B11.9515 images/s. This does not establish a native speedup.
Original corruption root cause remains unproven; current passing tests do not
establish that buffer lifetime caused the earlier failure.

Artifacts: `runs/hd5870-cast-revisit-20261101/` and
`runs/hd5870-workgroup-revisit-20261101/`. GPU-fault monitoring stops further
experiments after a captured reset/lockup/ring stall/failed CS relocation.
Each promising isolated change still needs full-model CPU comparisons and
matched native timing before promotion, followed by sustained validation.

Completed isolated workgroup matrix: all11 subprocesses exited0; kernel capture remained empty. No new defaults or full-model qualification at larger workgroups. Explicit gather/GEMM/index changes remain pending.

## HD6850 interim repeat

Barts PRO passed the same 183 top-k cases;772 byte comparisons and12 scatter
cases both with packing enabled and disabled; ten BN cases each at64/128/256;
99 copy comparisons and772 packed-byte comparisons each at automatic
caps32/64/128/256. Two cast-off batch24 CPU loss/gradient updates passed.
No kernel messages captured so far. Packed/native/packed timing is still
running its final control; do not infer a speedup from the first two trials.
Explicit gather/GEMM/index shapes and historical exact failure reproducers
are not qualified by these checks. Production defaults remain unchanged.
Artifacts: `runs/hd6850-workaround-revisit-20261101/`.

HD6850 repeat completed all21 stages successfully. Final packed/native/packed
batch24 throughput:7.5837/7.7609/8.0983 images/s; native is1.02% below the
control mean, inside the control spread. No established speed gain or regression.
Kernel capture remained empty. No default changes; no long candidate run.

## HD6850 matched workgroup/conversion optimization — completed

All six configurations passed two batch24 CPU loss/gradient updates each;
all12 native benchmarks completed with no new kernel messages. Frozen V9,
original comparison model, reservation960 MiB, queue128, retirement64 MiB,
one loader worker. Each timing used12 warmup and64 measured batches, with
configuration order reversed for the second repetition.

| Automatic cap / BN cap | Packed/widened path images/s | Native byte/int8 path images/s |
| --- | ---: | ---: |
| 32 / 64 (current) | 8.015 | 8.178 |
| 128 / 128 | 8.354 | 8.512 |
| 256 / 256 | 8.141 | 8.276 |

Means of two trials. Best128/native trials8.493 and8.531 versus control8.048
and7.982; best mean is6.20% above current control. Disabling packing at128
adds1.89%. Automatic cap is a maximum for unspecified local layouts, not a
claim every kernel uses128 threads. Explicit indexing/GEMM/gather geometry
was unchanged. BN and auto cap varied together, so their individual gains
are not isolated. Packed flag changes both int8 widening and byte storage;
this does not isolate casting cost or prove the old bug root cause.

Promising candidate, not sustained-run qualification. Production defaults
remain unchanged pending longer correctness/stability checks. Artifacts:
`runs/hd6850-workgroup-conversion-opt-20261101/`.

HD6850 candidate sustained run intentionally stopped by user after six added
epochs41-46 (about90minutes elapsed), aggregate8.5846training images/s.
49 TTM placement warnings; no captured resets/stalls. Last checkpoint loads
onCPU with finiteEMAweights and201 optimizerstates. This is a partial pass,
not completed20epoch stability qualification. Long per-card runs deferred
until the common minimal baseline is assembled. Defaults unchanged.

## HD6850 copy-cache flush disabled — targeted pass

With R600_CP_DMA_CB_FLUSH entirely unset, current auto128/BN128/packed0 V9
candidate passed100 native two-slice pool-growth trials (control also100),
99copy comparisons, queued temporary/lifetime stress,417pointwise offset
checks, and two B24 full-model CPU loss/gradient updates. No new kernel
messages. Per-kernel clFlush and deferred release remained enabled. CPU loss
max difference3.815e-5. No performance or sustained-run claim. Original
copy-flush failure was on6970/Cayman; targeted retest there remains required.
Production wrappers unchanged. Artifacts: runs/hd6850-copyflush-revisit-20261101/.

## HD6850 per-kernel flush removal — rejected

With candidate auto128/BN128/packed0 and both extra flush mechanisms OFF,
the native two-slice pool-growth reproducer fails trial0 with5120 wrong words.
All source/kernel/buffer objects survive completion in this fixture, so the
failure is not explained by premature Torch buffer destruction.
With per-kernel flush OFF but Mesa copy-cache flush ON, the native fixture
passes200trials across two fresh processes, but Torch's strided-copy suite
exitsSIGSEGV(-11) before reporting a passing case. No GPU reset/stall messages
captured. Segfault root cause remains unisolated.
Restoring per-kernel flush ON with copy-cache flush OFF passes all99 copy
comparisons again. Retain per-kernel flush in Barts candidate. Defaults unchanged.
No full-model test was run on the failing combinations. Artifacts:
`runs/hd6850-kernelflush-revisit-20261101/` and
`runs/hd6850-kernelflush-off-copyflush-on-20261101/`.

## HD6850 normal GEMM tiles — targeted pass

Both small/default tiles pass21 targeted cases each (8input gradients,
5full convolution autograd cases,4mm boundary cases,4historical M256/N1152/
K128 transpose variants). Two full-model B24CPU loss/gradient checks pass with
small tiles disabled. Four native ABBA probes,12warm/64timed each:
small8.52336/default8.56057/default8.52498/small8.52833 images/s.
Means small8.52584, default8.54277 (+0.20%); practically equivalent within
observed variation. No new kernel messages across7stages. This supports
DLPRIM_GEMM_SMALL_TILES=0 in the evolving Barts minimal candidate; no speedup
claim, sustained qualification or other-card claim. Production baseline wrappers
unchanged. Keep per-kernel flushON, copy-cacheflushOFF, auto128/BN128/packed0,
preallocation/deferred lifetime protection. Artifacts:
`runs/hd6850-gemmtiles-revisit-20261101/`.

## HD6850 Winograd and ordinary split-K — retain restrictions

Both independently abort during LLVM compilation (SIGABRT -6), not numerical
comparison. Winograd fails selecting AtomicCmpSwap in winconv_3x3_bwd_filter;
ordinary split-K fails selecting AtomicCmpSwap in sgemm on reduction-eligible
GEMMs. Current compiler cannot lower these operations in the tested path;
buffer lifetime changes cannot repair a compilation failure. Gather-input and
custom filter-gradient splitting remained enabled. These results do not rule
out a separately tested forward-only Winograd configuration.
Disabled-both control passed focused and2modelCPU updates, then native
benchmarks8.5399/8.5854images/s. TTM placement warnings occurred, no captured
GPUreset/stall. No enabled-candidate speed or full-model correctness claim.
Artifacts: `runs/hd6850-winograd-splitk-revisit-20261101/`.

## HD6850 explicit index cap / original copy layout — targeted pass

All16stages passed: controlindex32/flat1, index64/flat1, index128/flat1,
and index32/flat0. Each passed indexing/nonzero/bounds/gradient suite,
200sparse-mask repetitions,99copycomparisons and2B24modelCPUupdates.
Dispatch traces confirm prefix-scan widths32/64/128 and3D copy launches for
originalcopy. No newkernelmessages. Actual branchlessdivision/32bitmetadata
fixes retained; perkernelclFlush/preallocation/deferredretirement unchanged.
No timing, combinedindex128/flat0 test or sustainedqualification yet.
OriginalCayman failures still require targeted6970repeat. Defaultsunchanged.
Artifacts: `runs/hd6850-index-copy-revisit-20261101/`.

## HD6850 index/copy matched benchmark — completed

Combinedindex128/flat0 first passed indexing/gradient/bounds,sparse200,
copy99 and2B24CPUmodelupdates. All10 clean native trials passed; no new
kernelmessages. Each12warmup/64measured at B24,workers1,res960,V9,current
simplifiedcandidate. Forward and reverse configuration order. Means:
control32/flat1=8.54350;64/flat1=8.56498;128/flat1=8.61076;
32/flat0=8.58820;128/flat0=8.62600 images/s (+0.97% vscontrol).
Combined trials8.63627/8.61573 vscontrol8.51175/8.57525. Small measured
advantage, no statistical confidence interval or major speedup claim.
This supportsindex128/flat0 inBartsminimalcandidate, with arithmetic fixes,
perkernelflush andbufferprotectionsretained. Not sustained/othercardqualification.
Artifacts: `runs/hd6850-index-copy-benchmark-20261101/`.

## HD6850 original convolution/pooling backward — rejected

Separate gather overrides0 each abort duringLLVM AtomicCmpSwap selection:
convolution backward in sgemm; pooling backward in pooling_bw (SIGABRT-6).
Control passed21convolution/matrix cases,13pool/SPPF cases and2 B24modelCPU
updates under combinedindex128/flat0 minimalcandidate. Retain both gather
paths for this compiler/runtime. No failing-candidate benchmark attempted.
TTM placement warnings captured; no GPU reset/stall. This is a compiler-path
limitation, not evidence that every r600 hardware/runtime lacks atomics.
Artifacts: `runs/hd6850-gather-revisit-20261101/`.

## Cayman return — native bytes fail, packed control passes

2026-11-02: HD6970/Cayman1002:6718 on minimal-v1 stops at first nonempty
byte-comparison case:1/1wrong value. Sameprofile with ONLY packedoutputs1
passes all772checks. Largegroups/perkernelflush/poolretirement unchanged.
NoGPUreset/stall;179bytesofunrelatedAppArmor kernelmessage. This directly
supports a Cayman-specific byte workaround; lifetime changes did not eliminate
the observed failure here. Not proof of exact compiler/hardware cause.
Originalfailure andcontrol logs retained in
`runs/hd6970-minimal-v1-confirmation-20261101/`.

## Cayman BN128 hang reproduced — reboot needed

With packing1restored,772bytechecks andint8scatter pass. BN128 passes first8
module cases, then hangs before first large(2,4,128,128) case completes.
Livekernel reportsGPUlockup/ring0stalls; controller terminates tests andowned
monitor, butGPUremainsstalled. No moreGPUwork launched. Capture copied into
`runs/hd6970-minimal-v1-packed-confirmation-20261102/kernel-after-stall-snapshot.log`.
Next afterreboot: changeONLY BN128to64 while retaining packing1 andallother
candidate settings; confirmbefore proceeding. This reproduces historical
Cayman largerBN failure despite current lifetime/flush/preallocation profile.
It does not establish a universal128thread limit or failureofautomaticcap128.

## Cayman automatic cap32 — short confirmation passes

After updatedtopk testpassed and B28losschecks lookedclose, automaticcap128
training producedNaNs aroundbatch29, then scatterbounds erroraround34.
NoGPUhang orkernelmessages in thatfailure. ChangingONLYautomaticcap to32,
withpacking1/BN64/smalltiles1 andindex128/originalcopy retained, completes two
76-update nativeprobes,2CPUmodelupdates and800inferencechecks. No newkernel
messages. Training14.24551/14.20008i/s (mean14.22280), batch28/reserve1008MiB,
backendpeak1027617352bytes. CPUgradientcosines.9999998049/.9999998422
(vsbadcap128second.9999045). Decodedinference15.27752FPS, mean65.45565ms,
p95 65.78563ms. This is a short pass, not final sustainedqualification.

New namedCaymancandidate `R600_PROFILE=cayman-minimal-v1` records exceptions;
conservativeglobaldefault unchanged. Source/operator fixes common toallcards.
GPU RNG remainsknownunfixedLLVM15clauseoverflow; CPUinputtest change is not
proofthatgeneratorworks. Original source/layout/operatorcode unchanged by
this capcontrol. Exactcauseofauto128numericfailure remainsunknown, distinct
from RNGcompilerproof. Artifacts:
`runs/hd6970-minimal-v1-auto32-confirmation-20261102/` and failed128control
`runs/hd6970-minimal-v1-cpuinput-confirmation-20261102/`.

## Cypress confirmation: ordinary large GEMM hang

Minimal-v1 onHD5870/Cypress passes772nativebytechecks,scatter,BN128all10,
originalcopy,index128/sparse200. Conv/GEMMfixture passes8input-gradient and
5fullconvautogradcases,thenmm(1,2,3),(7,13,11),(33,17,65); nextmm
M256/N257/K128 hangs beforePASS. DefaultGEMMtiles selected; sourcegenerates
inputsCPUside. Exactdispatchuntraced(inputcopy/mm/readback withinfixture).
Kernelring0lockup/stalls persist; testcontroller stopped/no furtherGPUwork.
Afterreboot restoreONLYsmallGEMMtiles1, remainingCypresscandidateunchanged.
Do notgeneralizeBartsnormal-tilepass toCypress; noRNGcausalityevidencehere.
Artifacts: `runs/hd5870-minimal-v1-confirmation-20261102/`.

## Cypress small-tile control completes short confirmation

Afterfreshboot, changingONLYsmallGEMMtiles0to1 passesfull21casefixture,
remainingfocusedchecks,2 B24CPUmodelupdates,2native76updateprobes and800
inferencecomparisons. Priorbyte/BN128/copy/index128gateswerenotrepeated.
Training12.98685/13.24764i/s (mean13.11725), batch24,res960MiB;
peakbackend886266952bytes. Decoded17.40377FPS, mean57.4588ms,p95 60.9459ms.
No newkernelmessages. Final sustainedqualification remains deferred.
Cypresscandidateprofilecypress-minimal-v1 retains ONLYsmalltiles1 exception
relativeBartsminimal profile: auto/BN/index128,packed0,originalcopy,copycacheOFF
remain, plusperkernelflush/lifetimeprotection/gatherandalgorithmrestrictions.
No provencompilerclause-overflowcause assignedtothisGEMMhang.
Artifacts: `runs/hd5870-minimal-v1-smalltiles-confirmation-20261102/`.
