# [AMDGPU][r600/cayman] Oversized ALU clause is silently encoded with a wrapped count (LLVM 15)

## Summary

LLVM 15.0.6 targeting `r600` / `cayman` emits an ALU clause containing 130
64-bit slots, although the hardware ALU count field is seven bits and encodes
`slot_count - 1`. The internal count 129 is truncated to 1 in the ELF object,
so hardware sees two slots instead of 130. Both assembly and object generation
succeed without a diagnostic in the tested release build.

This can be reproduced entirely on a CPU using the attached 201-line LLVM IR.
No AMD GPU, Mesa, Clover, OpenCL runtime, PyTorch or DLPrimitives installation
is needed for the primary reproduction.

## Environment

- Debian LLVM version 15.0.6, optimized build (`llc-15`).
- Target triple `r600--`, CPU `cayman`.
- GPU used for independent execution evidence: Radeon HD 6970/Cayman XT,
  PCI 1002:6718, Linux 6.1.0-42-amd64.
- Execution runtime: Mesa 25.0.7 Clover/r600, with local pool-allocation/copy
  workarounds and per-kernel submission flushing. These runtime modifications
  are not required for the stock LLVM CPU-only encoding reproduction.
- I have not built/tested a newer LLVM release. Similar accounting code remains
  in current upstream source on inspection; that is not a newer-version repro.

## Reproduce without hardware

Unpack the attachment and run:

```sh
llc-15 -march=r600 -mcpu=cayman -filetype=asm repro.ll -o repro.s
llc-15 -march=r600 -mcpu=cayman -filetype=obj repro.ll -o repro.o
python3 inspect_alu.py repro.o
```

Alternatively run `bash reproduce.sh`. Its final exit 1 is intentional when the
inspector detects the invalid encoding. LLVM compilation itself exits 0.
The IR already includes/inlines the required builtins; no libclc input is needed.

Relevant generated assembly:

```text
ALU_PUSH_BEFORE 13, @8, ...
ALU 129, @22, ...
ALU 119, @152, ...
```

The middle clause physically spans addresses 22 through 151 inclusive:
`152 - 22 = 130` slots. The count operand is 129 (length minus one).

Inspector output on the generated object:

```text
CF[0] ALU address=8, encoded_count=13, encoded_slots=14, physical_span=14
CF[2] ALU address=22, encoded_count=1, encoded_slots=2, physical_span=130
CF[3] ALU address=152, encoded_count=119, encoded_slots=120, physical_span=120
INVALID clause length
```

The middle CF instruction contains word0 `0x80000016` and word1 `0xa0040000`.
Extracting count bits 18..24 gives 1, confirming actual object-code truncation,
rather than just a misleading assembly-printer value.

The included inspector is deliberately fixture-specific (one kernel, successive
ALU clauses); it is not a general r600 object validator.

## Expected behavior

Split clauses so their lengths fit the hardware encoding, or diagnose an
unencodable clause. Do not silently produce a truncated count and runnable
invalid code.

## Source-level lead, not a validated patch

The IR originated from an unrolled 10-round Philox uniform-random generator.
On Cayman, some integer multiplies expand into four instruction slots plus
literal payloads. `R600EmitClauseMarkers::OccupiedDwords()` returns 4 for vector
instructions before reaching literal counting. That appears to undercount
vector instructions with literal operands. In the machine-pass dump, the
relevant clause has estimated count 118 after clause marking, then final count
129 after expansion/packetization/literal materialization.

The finalizer sets the count from final clause contents. It has an assertion
against oversized clauses, but the tested release build continues to emit the
invalid encoding. This suggests two areas to investigate: correct early size
accounting, and robust validation before object emission. I have not validated
a compiler fix and am not proposing a workgroup-size change as the solution.

Relevant source:

- [R600 clause-size accounting, LLVM 15.0.6](https://github.com/llvm/llvm-project/blob/llvmorg-15.0.6/llvm/lib/Target/AMDGPU/R600EmitClauseMarkers.cpp)
- [R600 special-instruction expansion](https://github.com/llvm/llvm-project/blob/llvmorg-15.0.6/llvm/lib/Target/AMDGPU/R600ExpandSpecialInstrs.cpp)
- [Final literal insertion and count, LLVM 15.0.6](https://github.com/llvm/llvm-project/blob/llvmorg-15.0.6/llvm/lib/Target/AMDGPU/R600ControlFlowFinalizer.cpp#L380)
- [Mesa's documented seven-bit count encoding](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-25.0.7/src/gallium/drivers/r600/eg_sq.h)

## Independent GPU evidence (optional, not needed to reproduce encoding)

The stock CPU-generated `.text` section was byte-for-byte identical to the
cached native Clover kernel that hung the HD 6970. The isolated `fill` dispatch
returned `CL_SUCCESS`, then queue completion stalled and Linux reported ring 0
lockup. The initial automatic cap 128 selected local size 126, but a separate
native OpenCL fixture with explicit local 32 also hung.

That native fixture retained all buffers/programs/kernels until completion,
used a host-initialized guarded output buffer, and required only a 128 MiB driver
pool. It used 80640 output floats (divisible by four). There was no Torch,
YOLO, top-k, profiling, or previous batch-allocation history in that native test.
The input-length choice also avoids a separate non-multiple-of-four tail-store
issue in the original random source.

The optional `random_native.c` and original MIT-licensed `random.cl` are included
for context. GPU execution is not necessary for triage and can lock up the card,
requiring a reboot. Prefer the hardware-free IR reproduction above.

## Attachment contents

- `repro.ll`: linked/optimized LLVM IR, 201 lines.
- `reproduce.sh`: CPU-only compilation and inspection commands.
- `inspect_alu.py`: no-dependency ELF checker for this single-kernel fixture.
- `reproduction-output.txt`, `verification.json`, `llvm-version.txt`: recorded results.
- `random.cl`, `random_native.c`: original kernel and optional native execution fixture.
- `README.md`: provenance, usage and limitations.

No model weights, datasets, credentials or private machine paths are included.
