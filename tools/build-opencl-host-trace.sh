#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=${OCL_HOST_TRACE_SO:-"$repo_root/build-r600/opencl_host_trace.so"}
mkdir -p -- "$(dirname -- "$output")"
read -r -a compiler <<< "${CC:-cc}"
read -r -a cppflags <<< "${CPPFLAGS:-}"
read -r -a cflags <<< "${CFLAGS:--O2 -Wall -Wextra}"
read -r -a ldflags <<< "${LDFLAGS:-}"
read -r -a ldlibs <<< "${LDLIBS:--lOpenCL -ldl -pthread}"
"${compiler[@]}" "${cppflags[@]}" "${cflags[@]}" -shared -fPIC \
  "$repo_root/tools/opencl_host_trace.c" "${ldflags[@]}" "${ldlibs[@]}" -o "$output"
printf 'Built %s\n' "$output"
