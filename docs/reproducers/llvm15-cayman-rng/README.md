# LLVM15 r600/Cayman ALU-clause overflow

The primary reproduction is CPU only: no GPU, OpenCL loader, Mesa, Torch,
DLPrimitives installation or dataset is needed. Requires LLVM15 `llc` andPython3.
Run `bash reproduce.sh` (or `LLC=/path/to/llc bash reproduce.sh`). Exit1 from
`inspect_alu.py` means the invalid encoding was detected, not a harness error.
See `LLVM_CAYMAN_RNG_BUG.md` in the zip attachment (in the repository, it is two directories above this folder).

`repro.ll` was generated from Artyom Beilis's MIT-licensed DLPrimitives random
kernel with LLVM15/Clang15 and matching libclc, then internalized/inlined.
`random.cl` retains its upstream copyright/license. Its source flags were
`-DIS_UNIFORM=1 -DIS_NORMAL=0 -DIS_BERNOULLI=0`. Only LLVM IR is required for
the CPU reproduction; no builtins library is needed because it is already linked.

Optional hardware reproduction: `random_native.c` keeps all objects alive,
initializes a guarded output buffer, uses explicit workgroup sizes, and checks
all uniform outputs against a CPU implementation of the same Philox rounds.
Build with `cc -O2 -std=c11 random_native.c -lOpenCL -o random_native` using
OpenCL development headers and loader, select Clover/r600, then run
`./random_native 32 80640 random.cl`. On the testedCayman runtime this locks
up the GPU; a reboot is needed to recover. It is NOT necessary for confirming
the compiler encoding defect. Do not run it casually on a display GPU.

The inspector is deliberately limited to the supplied single-kernel fixture;
it infers physical clause spans from successive ALU addresses in this object.
It is not a general r600 binary validator. Never reuse its results for arbitrary
objects containing interspersed fetch clauses/multiple kernels without adapting it.
