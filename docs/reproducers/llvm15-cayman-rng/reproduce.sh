#!/usr/bin/env bash
# CPU-only LLVM reproduction. Exit1 from inspector means the bug was detected.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
compiler=${LLC:-llc-15}
"$compiler" -march=r600 -mcpu=cayman -filetype=asm repro.ll -o repro.s
"$compiler" -march=r600 -mcpu=cayman -filetype=obj repro.ll -o repro.o
python3 inspect_alu.py repro.o
