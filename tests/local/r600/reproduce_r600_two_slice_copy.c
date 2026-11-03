#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do{cl_int check_error=(c);if(check_error){fprintf(stderr,"%s: %d\n",#c,check_error);return 2;}}while(0)
int main(int argc,char**argv){
 int prime=argc>1&&!strcmp(argv[1],"prime");
 int fresh=argc>1&&!strcmp(argv[1],"fresh"),wait=argc>1&&!strcmp(argv[1],"wait");
 cl_platform_id p;cl_device_id d;cl_uint count;cl_int e;
 CHECK(clGetPlatformIDs(1,&p,&count));CHECK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,&count));cl_context ctx=clCreateContext(NULL,1,&d,NULL,NULL,&e);CHECK(e);cl_command_queue q=clCreateCommandQueue(ctx,d,0,&e);CHECK(e);
 const char*src="__kernel void prime(__global float*y,__global const float*a,__global const float*b){y[0]=a[0]+b[0];} __kernel void init(__global float*y,uint n,float v){uint i=get_global_id(0);if(i<n)y[i]=v+i;} __kernel void copy(ulong slice,ulong dim0,ulong tdim,ulong toff,ulong sdim,ulong soff,ulong dim2,__global float*y,ulong yo,__global const float*x,ulong xo,float scale){ulong p0=get_global_id(0),p1=get_global_id(1),p2=get_global_id(2);if(p0>=dim2||p1>=slice||p2>=dim0)return;y+=yo+p0+(p1+toff)*dim2+p2*tdim*dim2;x+=xo+p0+(p1+soff)*dim2+p2*sdim*dim2;if(scale==0)*y=*x;else *y=scale**y+*x;}";
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,NULL,&e);CHECK(e);CHECK(clBuildProgram(pr,1,&d,"",NULL,NULL));cl_kernel init=clCreateKernel(pr,"init",&e);CHECK(e);
 cl_uint n=2*1344*2;float*host=malloc(2*n*4);if(!host)return 2;size_t ig=(n+63)/64*64,il=64;size_t global[3]={1,2,2688};
 for(unsigned trial=0;trial<100;trial++){
  cl_mem inputs[2],y;
  for(int s=0;s<2;s++){
   inputs[s]=clCreateBuffer(ctx,CL_MEM_READ_WRITE,32768,NULL,&e);CHECK(e);float v=s?1000000.f:1000.f;
   CHECK(clSetKernelArg(init,0,sizeof(cl_mem),&inputs[s]));CHECK(clSetKernelArg(init,1,4,&n));CHECK(clSetKernelArg(init,2,4,&v));CHECK(clEnqueueNDRangeKernel(q,init,1,NULL,&ig,&il,0,NULL,NULL));CHECK(clFinish(q));CHECK(clEnqueueReadBuffer(q,inputs[s],CL_TRUE,0,n*4,host,0,NULL,NULL));
   for(unsigned i=0;i<n;i++)if(host[i]!=v+i){printf("INITFAIL %d %u\n",s,i);return 1;}
  }
  y=clCreateBuffer(ctx,CL_MEM_READ_WRITE,65536,NULL,&e);CHECK(e);
  if(prime){cl_kernel pk=clCreateKernel(pr,"prime",&e);CHECK(e);CHECK(clSetKernelArg(pk,0,sizeof(y),&y));CHECK(clSetKernelArg(pk,1,sizeof(inputs[0]),&inputs[0]));CHECK(clSetKernelArg(pk,2,sizeof(inputs[1]),&inputs[1]));size_t one=1;CHECK(clEnqueueNDRangeKernel(q,pk,1,NULL,&one,&one,0,NULL,NULL));CHECK(clFinish(q));clReleaseKernel(pk);}
  cl_kernel kernels[2];kernels[0]=clCreateKernel(pr,"copy",&e);CHECK(e);kernels[1]=fresh?clCreateKernel(pr,"copy",&e):kernels[0];CHECK(e);
  for(int s=0;s<2;s++){
   cl_kernel k=kernels[s];cl_ulong params[7]={2,2688,4,s*2,2,0,1},offset=0;float scale=0;
   for(int a=0;a<7;a++)CHECK(clSetKernelArg(k,a,8,&params[a]));CHECK(clSetKernelArg(k,7,sizeof(y),&y));CHECK(clSetKernelArg(k,8,8,&offset));CHECK(clSetKernelArg(k,9,sizeof(cl_mem),&inputs[s]));CHECK(clSetKernelArg(k,10,8,&offset));CHECK(clSetKernelArg(k,11,4,&scale));CHECK(clEnqueueNDRangeKernel(q,k,3,NULL,global,NULL,0,NULL,NULL));if(wait)CHECK(clFinish(q));
  }
  CHECK(clFinish(q));CHECK(clEnqueueReadBuffer(q,y,CL_TRUE,0,n*8,host,0,NULL,NULL));
  unsigned bad=0;for(unsigned i=0;i<2*n;i++){unsigned row=i/4,c=i%4;float expected=(c<2?1000.f:1000000.f)+row*2+c%2;if(host[i]!=expected){if(bad<3)printf("BAD trial%u word%u expected%.0f actual%.0f\n",trial,i,expected,host[i]);bad++;}}
  printf("trial%u fresh%d wait%d bad%u\n",trial,fresh,wait,bad);fflush(stdout);if(bad)return 1;
  clReleaseKernel(kernels[0]);if(fresh)clReleaseKernel(kernels[1]);clReleaseMemObject(inputs[0]);clReleaseMemObject(inputs[1]);clReleaseMemObject(y);
 }
 puts("PASS100");return 0;
}
