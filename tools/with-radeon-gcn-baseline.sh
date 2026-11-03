#!/usr/bin/env bash
# GCN control profile: ordinary buffers, no r600 global-pool/launch workarounds.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (($# == 0)); then echo 'Usage: with-radeon-gcn-baseline.sh COMMAND [ARG ...]' >&2; exit 2; fi
backend=${DLPRIM_PYTHON_PATH:-"$repo_root/build"}
if [[ ! -f "$backend/pytorch_ocl/pt_ocl.so" ]]; then echo 'Build or select DLPRIM_PYTHON_PATH first' >&2; exit 2; fi
unset LD_PRELOAD R600_POOL_RESERVE_MB R600_MESA_PREFIX GALLIUM_PIPE_SEARCH_DIR
unset R600_PROFILE R600_AUTO_WG_MAX R600_CP_DMA_CB_FLUSH R600_FLUSH_AFTER_KERNEL
for flag in ${!DLPRIM_@}; do
  if [[ $flag != DLPRIM_PYTHON_PATH ]]; then unset "$flag"; fi
done
export PYTHONPATH="$backend${PYTHONPATH:+:$PYTHONPATH}"
export OPENCL_NO_MEM_CACHE=1 OPENCL_TIGHT_ALLOC=1 OPENCL_DEFERRED_RELEASE=1
export OPENCL_ASYNC_OP_LIMIT=${OPENCL_ASYNC_OP_LIMIT:-128}
export OPENCL_DEFERRED_RELEASE_MB=${OPENCL_DEFERRED_RELEASE_MB:-64}
exec "$@"
