# Cayman BatchNorm transition investigation — 2026-11-02

Scope: source inspection and existing captures, not a new GPU experiment.
Source: DLPrimitives `src/core/bn.cpp`, `src/kernels/bn_sums.cl`,
`src/kernels/reduce2.h`. Current GPU stalled later in a separate convolution
fixture; do not run more GPU work before reboot.

## Observations

Cayman BN128 passed eight module cases, then hung before first large
(2,4,128,128) case completed. Kernel reports ring0stalled/GPUlockup.
After reboot, changing ONLY BN cap128to64 (packing1retained) passes all ten
cases including both affine/nonaffine large cases. Automatic cap remains128;
BN uses explicit/required local sizes, so the shim does not replace them.
This is not evidence all128-thread launches fail onCayman.

## Exact transition in current source

`total = tensor_elements/channels = batch*spatial`; then
`second_size = ceil(total/256)`. Single stage when second_size<64; otherwise
partial reduction plus final reduction. Therefore the first threshold is
**16129 elements per channel**, not exactly16384. Selection of partial count
uses second_size thresholds64/128/256, then the same BN cap clamps BOTH the
first-stage workgroup and the number of partial groups/final-stage width.
The flag changes more than one launch parameter.

For failing input batch2,channels4,HW16384: total32768, second_size128.

| Quantity | BN128 (hang) | BN64 (pass) |
| --- | ---: | ---: |
| Input tensor bytes | 524288 | 524288 |
| First-stage local shape | 128x1 | 64x1 |
| First-stage global shape | 16384x4 | 4096x4 |
| Partial groups per channel | 128 | 64 |
| Samples per first-stage lane | 2 | 8 |
| Final-stage global shape | 128x4 | 64x4 |
| Final-stage local shape | 128x1 | 64x1 |
| Two local FP32 reduction planes per group | 1024 bytes | 512 bytes |
| Total advertised BN workspace | 4176 bytes | 2128 bytes |

First-stage `compute` writes two partial arrays (sum and squared sum in
forward, dy*x and dy in backward); final `reduce` merges them. Host uses
series contexts/events between dispatches; both source kernels reduce two
local-memory planes with a barrier at every tree level. All lanes of a group
have the same feature ID and exact feature-sizedY launch; the early channel
return is therefore uniform per group under this dispatch. This inspection
has not proved every bounds, compiler or synchronization property correct.

No obvious GiB-scale allocation in this transition. Same1008MiB pool
reservation works forBN64. Workspace changes by2KiB, so an allocation-limit
explanation is weak, but placement effects are not formally excluded.

## Next diagnostic if deeper root-cause work is authorized

Use a standalone native OpenCL fixture on a fresh boot, without changing
production kernels:

1. Capture compiled macros, required/actual local/global sizes and kernel args.
2. Run first-stage128 alone, keep every buffer alive, finish/read partials and
   compare against CPU. Stop immediately on fault.
3. In a separate fresh-process test, run final-stage128 with known host-generated
   partials, independently of first-stage output. Compare againstCPU.
4. If isolated stages pass, test chain with explicit completion between stages,
   then original event-ordered chain. Also test a small initial pool reservation
   to separate driver placement from execution geometry.
5. Log enqueue versus completion IDs so a hang identifies a particular dispatch,
   rather than only the enclosing PyTorch operation. Only after isolation inspect
   generated LLVM/ISA or file an upstream compiler/driver issue.

Do not introduce unsafe thread-shape overrides for kernels with
`reqd_work_group_size`; compile matching definitions. Do not call this an
int8-counter bug, a proven barrier defect, or faulty physicalGPU memory.
KeepBN64 in the candidate while gathering evidence. This diagnostics plan is
not queued or running and does not expand the baseline pass into new tuning.

## Separate later failure

On the BN64 boot, bytes/scatter, original copies, indexing128 and sparse200
pass. The convolution/GEMM suite passes8 input-gradient and3 small full-autograd
cases, then hangs before the first large exact/fringe convolution reports PASS
(ci128/co256,stride2,16x16). Normal GEMM tiles were selected, but no dispatch
trace identifies which forward/backward kernel hung. Revert small-tile override
ON as the next SINGLE controlled change after reboot, rather than declaring
normaltiles universally safe or blaming the already-passing BN64 case.

Artifacts: `runs/hd6970-minimal-v1-packed-confirmation-20261102/` and
`runs/hd6970-minimal-v1-packed-bn64-confirmation-20261102/`.
