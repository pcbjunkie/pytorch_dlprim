#!/usr/bin/env bash
# Explicit opt-in profile. See docs/r600-clover.md before selecting a device.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
backend=${DLPRIM_PYTHON_PATH:-"$repo_root/build"}
shim=${R600_SHIM:-"$repo_root/build-r600/r600_auto_workgroups.so"}
if (($# == 0)); then
  echo 'Usage: tools/with-r600-clover.sh COMMAND [ARG ...]' >&2
  exit 2
fi
if [[ ! -f "$shim" || ! -f "$backend/pytorch_ocl/pt_ocl.so" ]]; then
  echo 'Build the backend and tools/build-r600-shim.sh first; see docs/r600-clover.md.' >&2
  exit 2
fi
if [[ -z ${OCL_ICD_VENDORS:-} ]]; then
  echo 'Select an isolated patched Clover ICD directory using OCL_ICD_VENDORS.' >&2
  exit 2
fi
if [[ -n ${R600_MESA_PREFIX:-} ]]; then
  export LD_LIBRARY_PATH="$R600_MESA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export GALLIUM_PIPE_SEARCH_DIR=${GALLIUM_PIPE_SEARCH_DIR:-"$R600_MESA_PREFIX/lib/gallium-pipe"}
fi
export PYTHONPATH="$backend${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD="$shim${LD_PRELOAD:+:$LD_PRELOAD}"
export DLPRIM_FLAT_STRIDED_COPY=1 DLPRIM_INDEX_WG=32
export DLPRIM_CONV_GATHER=1 DLPRIM_POOL_GATHER=1
export DLPRIM_BN_MAX_WG=64 DLPRIM_GEMM_SMALL_TILES=1
export DLPRIM_PACKED_BYTE_OUTPUTS=1 DLPRIM_DISABLE_WINOGRAD=1 DLPRIM_DISABLE_SPLIT_K=1
export DLPRIM_FILTER_SPLIT_K=1 DLPRIM_INDEX_SORT_REDUCE=1
export DLPRIM_FUSED_SILU_BACKWARD=1 DLPRIM_GATHER_VECTOR4=1
export DLPRIM_GATHER_SPATIAL4=1 DLPRIM_GATHER_SMALL1X1=1
export OPENCL_NO_MEM_CACHE=1 OPENCL_TIGHT_ALLOC=1 OPENCL_DEFERRED_RELEASE=1
export OPENCL_ASYNC_OP_LIMIT=${OPENCL_ASYNC_OP_LIMIT:-32}
export OPENCL_DEFERRED_RELEASE_MB=${OPENCL_DEFERRED_RELEASE_MB:-32}
export DLPRIM_FILTER_SPLIT_GROUPS=${DLPRIM_FILTER_SPLIT_GROUPS:-256}
# Deliberately no default: an established reserve on 2 GB is unsuitable for 1 GB.
export R600_CP_DMA_CB_FLUSH=1
exec "$@"
