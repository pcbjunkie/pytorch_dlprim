#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=${R600_FLUSH_SHIM:-"$repo_root/build-r600/r600_flush_after_kernel.so"}
mkdir -p -- "$(dirname -- "$output")"
read -r -a compiler <<< "${CC:-cc}"
read -r -a cppflags <<< "${CPPFLAGS:-}"
read -r -a cflags <<< "${CFLAGS:--O2}"
read -r -a ldflags <<< "${LDFLAGS:-}"
read -r -a ldlibs <<< "${LDLIBS:--lOpenCL -ldl}"
"${compiler[@]}" "${cppflags[@]}" "${cflags[@]}" -shared -fPIC \
  "$repo_root/tools/r600_flush_after_kernel.c" "${ldflags[@]}" "${ldlibs[@]}" -o "$output"
printf 'Built %s\n' "$output"
