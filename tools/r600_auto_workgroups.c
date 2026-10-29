#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <dlfcn.h>
#include <stdio.h>
static unsigned changed;
cl_int clEnqueueNDRangeKernel(cl_command_queue q,cl_kernel k,cl_uint dims,const size_t*offset,const size_t*global,const size_t*local,cl_uint ne,const cl_event*events,cl_event*event){
 cl_int(*real)(cl_command_queue,cl_kernel,cl_uint,const size_t*,const size_t*,const size_t*,cl_uint,const cl_event*,cl_event*)=dlsym(RTLD_NEXT,"clEnqueueNDRangeKernel");
 if(local||dims<1||dims>3)return real(q,k,dims,offset,global,local,ne,events,event);
 cl_device_id device;size_t req[3]={0,0,0},maximum=32;
 if(clGetCommandQueueInfo(q,CL_QUEUE_DEVICE,sizeof(device),&device,NULL) || clGetKernelWorkGroupInfo(k,device,CL_KERNEL_COMPILE_WORK_GROUP_SIZE,sizeof(req),req,NULL) || req[0])return real(q,k,dims,offset,global,local,ne,events,event);
 size_t allowed;if(!clGetKernelWorkGroupInfo(k,device,CL_KERNEL_WORK_GROUP_SIZE,sizeof(allowed),&allowed,NULL) && allowed<maximum)maximum=allowed;
 size_t best[3]={1,1,1},product=1;
 for(size_t x=1;x<=maximum;x++)if(global[0]%x==0)
  for(size_t y=1;y<=(dims>1?maximum/x:1);y++)if(dims<2||global[1]%y==0)
   for(size_t z=1;z<=(dims>2?maximum/(x*y):1);z++)if((dims<3||global[2]%z==0)&&x*y*z>=product){product=x*y*z;best[0]=x;best[1]=y;best[2]=z;}
 if(__sync_add_and_fetch(&changed,1)==1)fprintf(stderr,"AUTO_WG_LIMIT32_XFIRST enabled for unspecified local sizes; compiled required sizes unchanged\n");
 return real(q,k,dims,offset,global,best,ne,events,event);
}
