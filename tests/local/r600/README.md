# Experimental r600 OpenCL correctness reproducers

These standalone OpenCL 1.2 programs need a C compiler, OpenCL headers/loader,
and a working GPU OpenCL ICD. They do not need Torch, YOLO, datasets or network
access. Explicitly select the intended ICD/device before comparing results;
each program uses the first platform's first GPU.

Example build on a standard development installation:

```sh
cc -O2 reproduce_r600_two_slice_copy.c -lOpenCL -o two_slice
./two_slice reuse
./two_slice fresh
./two_slice wait
./two_slice prime
```

Programs return0 for a completed successful run,1 for a mismatch,2 for an API
error. A timeout is not a completed pass. Keep logs and record exact driver,
compiler, GPU, launch dimensions and wrapper flags for any report.

## Two-slice shared-pool ordering

`reproduce_r600_two_slice_copy.c`: two verified inputs are concatenated using
back-to-back slice-copy dispatches. `fresh` uses separate kernel objects; `wait`
finishes after each copy; `prime` pre-promotes all buffers with a preliminary
kernel. Cayman reproduces corruption immediately in reuse/fresh; wait/prime pass
100 repetitions. A diagnostic driver flush before pool promotion/copy clears
this particular reproducer. This does not establish all training is reliable.

## Sparse coordinates

`reproduce_r600_nonzero_coords.c`: CPU supplies a fixed20-element sparse mask,
correct prefix counts and block offsets. GPU only computes coordinates. Optional
argument is local workgroup size (default64).

`reproduce_r600_nonzero_coords_branchless.c` uses equivalent branchless restoring
division. Original local64 failed at27; branchless local64 passed2000. Original
local32 also passed200. Thus the branchless change is an observed workaround,
not proof of a particular LLVM defect. The API-equivalent Torch regression is
`../check_nonzero_sparse_mask.py`.

## Strided copies with independent GPU verification

`reproduce_r600_strided_copy_verify.c` copies a [2,1344,64] view with underlying
source shape [2,69,1344]. Argument `explicit` selects local[64,1,1], otherwise
runtime chooses. A second kernel scans source/output before host readback.
Both GPU/CPU detected a wrong output at11685 in one test.

`reproduce_r600_strided_copy_flat5.c` uses a flattened1D dispatch for the
[2,1344,5] view, source offset86016. `explicit 64` failed at iteration2;
`explicit 32` completed250 GPU/CPU checks. Its workgroup size can be selected
with the second argument. `reproduce_r600_strided_copy_flat5_source.c` traverses
source-contiguous order instead; explicit64 completed250.

Failing isolated work-item IDs1125,11685,3685 and11877 are all37 modulo64.
This lane dependence does not determine whether the ultimate cause is driver,
compiler or hardware. Testing another card/driver is needed for that attribution.
The backend's experimental flat-copy/WG32 profile is being validated separately.
Never treat short reproducer passes as a full-training guarantee.
