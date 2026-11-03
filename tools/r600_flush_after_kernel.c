#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <dlfcn.h>
#include <stdio.h>

/* Opt-in driver workaround. Submit each dispatch without a host-side finish.
 * Loaded before the separate conservative workgroup shim by the profile.
 */
cl_int clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel kernel,
    cl_uint dimensions, const size_t *offset, const size_t *global,
    const size_t *local, cl_uint count, const cl_event *dependencies,
    cl_event *event)
{
    static unsigned announced;
    cl_int (*next)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
        const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) =
        dlsym(RTLD_NEXT, "clEnqueueNDRangeKernel");
    cl_int result = next(queue, kernel, dimensions, offset, global, local,
                         count, dependencies, event);
    if (result == CL_SUCCESS) {
        if (!__sync_fetch_and_add(&announced, 1))
            fprintf(stderr, "R600_FLUSH_AFTER_KERNEL enabled; no host finish inserted\n");
        result = clFlush(queue);
    }
    return result;
}
