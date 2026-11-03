#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CHECK(x) do{cl_int z=(x);if(z){fprintf(stderr,"API_ERROR line%d code%d\n",__LINE__,z);return 2;}}while(0)
typedef struct{uint32_t l0,r0,l1,r1,k0,k1;}state;
static void expected(uint64_t seed,uint64_t seq,float*out){state s={0,0,(uint32_t)(seq>>32),(uint32_t)seq,(uint32_t)seed,(uint32_t)(seed>>32)};
 for(int i=0;i<10;i++){state n;n.l1=(uint32_t)((uint64_t)s.r1*0xD2511F53u);n.r1=(uint32_t)(((uint64_t)s.r0*0xCD9E8D57u)>>32)^s.k0^s.l0;n.l0=(uint32_t)((uint64_t)s.r0*0xCD9E8D57u);n.r0=(uint32_t)(((uint64_t)s.r1*0xD2511F53u)>>32)^s.k1^s.l1;n.k0=s.k0+0xBB67AE85u;n.k1=s.k1+0x9E3779B9u;s=n;}
 out[0]=(s.l0>>8)*0x1p-24f;out[1]=(s.r0>>8)*0x1p-24f;out[2]=(s.l1>>8)*0x1p-24f;out[3]=(s.r1>>8)*0x1p-24f;}
int main(int argc,char**argv){setvbuf(stdout,0,_IONBF,0);if(argc!=4){fprintf(stderr,"usage: probe LOCAL ELEMENTS SOURCE\n");return 2;}size_t local=strtoul(argv[1],0,10);cl_ulong total=strtoull(argv[2],0,10);if(!local||!total||total>1000000)return 2;
 FILE*f=fopen(argv[3],"rb");if(!f)return 2;fseek(f,0,SEEK_END);size_t size=ftell(f);rewind(f);char*source=calloc(size+1,1);if(fread(source,1,size,f)!=size)return 2;fclose(f);
 cl_platform_id platform;cl_device_id dev;cl_uint count;cl_int error;CHECK(clGetPlatformIDs(1,&platform,&count));CHECK(clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&dev,&count));
 cl_context ctx=clCreateContext(0,1,&dev,0,0,&error);CHECK(error);cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&error);CHECK(error);
 cl_program program=clCreateProgramWithSource(ctx,1,(const char**)&source,&size,&error);CHECK(error);error=clBuildProgram(program,1,&dev,"-DIS_UNIFORM=1 -DIS_NORMAL=0 -DIS_BERNOULLI=0",0,0);if(error){char log[16384]={0};clGetProgramBuildInfo(program,dev,CL_PROGRAM_BUILD_LOG,sizeof(log),log,0);fprintf(stderr,"BUILD_ERROR %s\n",log);return 2;}
 cl_kernel kernel=clCreateKernel(program,"fill",&error);CHECK(error);
 size_t length=(size_t)total+32;float*host=malloc(length*sizeof(float));for(size_t i=0;i<length;i++)host[i]=-12345.5f;
 cl_mem buffer=clCreateBuffer(ctx,CL_MEM_READ_WRITE,length*sizeof(float),0,&error);CHECK(error);CHECK(clEnqueueWriteBuffer(q,buffer,CL_TRUE,0,length*sizeof(float),host,0,0,0));
 cl_ulong offset=16,seed=12345,seq=7;float low=0,high=1;CHECK(clSetKernelArg(kernel,0,sizeof(total),&total));CHECK(clSetKernelArg(kernel,1,sizeof(buffer),&buffer));CHECK(clSetKernelArg(kernel,2,sizeof(offset),&offset));CHECK(clSetKernelArg(kernel,3,sizeof(seed),&seed));CHECK(clSetKernelArg(kernel,4,sizeof(seq),&seq));CHECK(clSetKernelArg(kernel,5,sizeof(low),&low));CHECK(clSetKernelArg(kernel,6,sizeof(high),&high));
 size_t global=((((size_t)total+3)/4+local-1)/local)*local;printf("NATIVE_ENQUEUE total%llu global%zu local%zu\n",(unsigned long long)total,global,local);CHECK(clEnqueueNDRangeKernel(q,kernel,1,0,&global,&local,0,0,0));printf("NATIVE_FINISH_BEGIN\n");CHECK(clFinish(q));printf("NATIVE_FINISH_PASS\n");CHECK(clEnqueueReadBuffer(q,buffer,CL_TRUE,0,length*sizeof(float),host,0,0,0));
 unsigned bad=0,guards=0;for(size_t i=0;i<length;i++){if(i<16||i>=16+(size_t)total){if(host[i]!=-12345.5f){if(guards<8)printf("GUARD_BAD %zu %.9g\n",i,host[i]);guards++;}}}
 for(size_t i=0;i<(size_t)total;i+=4){float ref[4];expected(seed,seq+i/4,ref);for(size_t j=0;j<4&&i+j<(size_t)total;j++)if(host[16+i+j]!=ref[j]){if(bad<8)printf("VALUE_BAD %zu %.9g %.9g\n",i+j,host[16+i+j],ref[j]);bad++;}}
 printf("NATIVE_RESULT value_errors%u guard_errors%u\n",bad,guards);clReleaseKernel(kernel);clReleaseProgram(program);clReleaseMemObject(buffer);clReleaseCommandQueue(q);clReleaseContext(ctx);free(host);free(source);return bad||guards?1:0;}
