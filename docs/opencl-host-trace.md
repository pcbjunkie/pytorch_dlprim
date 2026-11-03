# OpenCL host API timing helper

`tools/opencl_host_trace.c` is an optional diagnostic preload library. Build it
with `tools/build-opencl-host-trace.sh`; `CC`, `CPPFLAGS`, `CFLAGS`, `LDFLAGS`,
`LDLIBS` and `OCL_HOST_TRACE_SO` can override compilation and the output path.
The default output is `build-r600/opencl_host_trace.so`. OpenCL headers, a C
compiler, the OpenCL loader, libdl and pthreads are needed.

Add the library to the profiling process's `LD_PRELOAD`, set
`OCL_HOST_TRACE_FILE` to a writable CSV filename, and start with
`OCL_HOST_TRACE_ACTIVE=0`. The existing r600 wrapper preserves a caller's
preload libraries after its optional workaround shims. This ordering records
the underlying enqueue call and the per-kernel flush as separate API calls.

Inside a bounded, synchronized diagnostic region, set
`os.environ['OCL_HOST_TRACE_ACTIVE']='1'`, run the selected work, synchronize,
then restore it to `'0'`. Flush the buffered CSV with
`ctypes.CDLL(None).ocl_host_trace_flush()` after the window. The file is also
closed at process exit. Use a distinct output file for each process.

The CSV records monotonic start/end nanoseconds, API/kernel name, blocking
flag, requested bytes and OpenCL return status. It covers NDRange enqueue,
buffer reads/writes/copies, flush, finish, event waits, marker enqueue, buffer
creation and release. It does not add flushes, finishes or waits and preserves
the called API's result. It is not a complete trace of every driver operation.

Kernel-name queries, clocks, environment checks, locking and buffered file
writes perturb timings. Never load this helper for the native throughput
baseline. Nested API calls can overlap: calculate interval unions before
reporting total time inside APIs. Explicit waits can overlap device execution,
and time inside `clFlush` can include driver stalls; do not add API time to GPU
event time or call it pure CPU computation. GPU-event and host timestamp
origins also differ unless explicitly calibrated.

The [HD5870 study](hd5870-utilization.md) keeps diagnostic API/event captures
separate from native training and inference sweeps. Its raw local analyzer
uses interval unions and retains the profiling limitations in the exported JSON.
