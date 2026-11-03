#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Diagnostic host API intervals only. No added queue flushes, waits or finishes.
 * Activate only inside the profiled window with OCL_HOST_TRACE_ACTIVE=1.
 * Calls can be nested; consumers must use interval unions, not sum all rows.
 */
static pthread_mutex_t trace_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *trace_file;

static int active(void)
{
    const char *value = getenv("OCL_HOST_TRACE_ACTIVE");
    return value && value[0] == '1';
}

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}

static void emit(uint64_t start, const char *name, int blocking,
                 size_t bytes, cl_int status)
{
    uint64_t end = now_ns();
    pthread_mutex_lock(&trace_lock);
    if (!trace_file) {
        const char *path = getenv("OCL_HOST_TRACE_FILE");
        if (path) {
            trace_file = fopen(path, "w");
            if (trace_file) {
                setvbuf(trace_file, NULL, _IOFBF, 65536);
                fprintf(trace_file, "start_ns,end_ns,api,blocking,bytes,status\n");
            }
        }
    }
    if (trace_file)
        fprintf(trace_file, "%" PRIu64 ",%" PRIu64 ",%s,%d,%zu,%d\n",
                start, end, name, blocking, bytes, status);
    pthread_mutex_unlock(&trace_lock);
}

void ocl_host_trace_flush(void)
{
    pthread_mutex_lock(&trace_lock);
    if (trace_file)
        fflush(trace_file);
    pthread_mutex_unlock(&trace_lock);
}

__attribute__((destructor)) static void close_trace(void)
{
    if (trace_file)
        fclose(trace_file);
}

#define RESOLVE(name) static __thread fn next; \
    if (!next) next = (fn)dlsym(RTLD_NEXT, #name)
#define START int tracing = active(); uint64_t start = tracing ? now_ns() : 0
#define END(name,blocking,bytes) if (tracing) emit(start,name,blocking,bytes,result)

cl_int clEnqueueNDRangeKernel(cl_command_queue q, cl_kernel k, cl_uint d,
    const size_t *o, const size_t *g, const size_t *l, cl_uint n,
    const cl_event *wait, cl_event *event)
{
    typedef cl_int (*fn)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
        const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
    RESOLVE(clEnqueueNDRangeKernel);
    char name[256] = "kernel:unknown";
    int tracing = active();
    if (tracing)
        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(name)-7, name+7, NULL);
    uint64_t start = tracing ? now_ns() : 0;
    cl_int result = next(q,k,d,o,g,l,n,wait,event);
    END(name,0,0);
    return result;
}

cl_int clEnqueueReadBuffer(cl_command_queue q, cl_mem m, cl_bool blocking,
    size_t offset, size_t bytes, void *ptr, cl_uint n,
    const cl_event *wait, cl_event *event)
{
    typedef cl_int (*fn)(cl_command_queue,cl_mem,cl_bool,size_t,size_t,void*,cl_uint,const cl_event*,cl_event*);
    RESOLVE(clEnqueueReadBuffer); START;
    cl_int result = next(q,m,blocking,offset,bytes,ptr,n,wait,event);
    END("read_buffer",blocking,bytes); return result;
}

cl_int clEnqueueWriteBuffer(cl_command_queue q, cl_mem m, cl_bool blocking,
    size_t offset, size_t bytes, const void *ptr, cl_uint n,
    const cl_event *wait, cl_event *event)
{
    typedef cl_int (*fn)(cl_command_queue,cl_mem,cl_bool,size_t,size_t,const void*,cl_uint,const cl_event*,cl_event*);
    RESOLVE(clEnqueueWriteBuffer); START;
    cl_int result = next(q,m,blocking,offset,bytes,ptr,n,wait,event);
    END("write_buffer",blocking,bytes); return result;
}

cl_int clEnqueueCopyBuffer(cl_command_queue q, cl_mem src, cl_mem dst,
    size_t a, size_t b, size_t bytes, cl_uint n, const cl_event *wait, cl_event *event)
{
    typedef cl_int (*fn)(cl_command_queue,cl_mem,cl_mem,size_t,size_t,size_t,cl_uint,const cl_event*,cl_event*);
    RESOLVE(clEnqueueCopyBuffer); START;
    cl_int result = next(q,src,dst,a,b,bytes,n,wait,event);
    END("copy_buffer",0,bytes); return result;
}

cl_int clFlush(cl_command_queue q)
{
    typedef cl_int (*fn)(cl_command_queue); RESOLVE(clFlush); START;
    cl_int result = next(q); END("flush",0,0); return result;
}

cl_int clFinish(cl_command_queue q)
{
    typedef cl_int (*fn)(cl_command_queue); RESOLVE(clFinish); START;
    cl_int result = next(q); END("finish",1,0); return result;
}

cl_int clWaitForEvents(cl_uint n, const cl_event *events)
{
    typedef cl_int (*fn)(cl_uint,const cl_event*); RESOLVE(clWaitForEvents); START;
    cl_int result = next(n,events); END("wait_events",1,0); return result;
}

cl_int clEnqueueMarkerWithWaitList(cl_command_queue q, cl_uint n,
    const cl_event *wait, cl_event *event)
{
    typedef cl_int (*fn)(cl_command_queue,cl_uint,const cl_event*,cl_event*);
    RESOLVE(clEnqueueMarkerWithWaitList); START;
    cl_int result = next(q,n,wait,event); END("marker",0,0); return result;
}

cl_mem clCreateBuffer(cl_context c, cl_mem_flags flags, size_t bytes,
    void *ptr, cl_int *error)
{
    typedef cl_mem (*fn)(cl_context,cl_mem_flags,size_t,void*,cl_int*);
    RESOLVE(clCreateBuffer); START;
    cl_mem memory = next(c,flags,bytes,ptr,error);
    cl_int result = error ? *error : (memory ? CL_SUCCESS : CL_OUT_OF_RESOURCES);
    END("create_buffer",0,bytes); return memory;
}

cl_int clReleaseMemObject(cl_mem memory)
{
    typedef cl_int (*fn)(cl_mem); RESOLVE(clReleaseMemObject); START;
    cl_int result = next(memory); END("release_buffer",0,0); return result;
}
