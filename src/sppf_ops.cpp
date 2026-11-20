#include "utils.h"
#include <map>
#include <mutex>
namespace ptdlprim {
using namespace at;
static cl::Program sppf_program(dlprim::Context &ctx) {
 static std::mutex mutex;
 static std::map<std::pair<cl_context,cl_device_id>,cl::Program> programs;
 std::lock_guard<std::mutex> lock(mutex);
 auto key=std::make_pair(ctx.context()(),ctx.device()());auto it=programs.find(key);
 if(it!=programs.end())return it->second;
 const char *source=R"cl(
 inline float pool_max(float a,float b){return isnan(a)?a:(isnan(b)?b:fmax(a,b));}
 __kernel void sppf_row(__global const float *input,ulong offset,__global float *out,__global float *temp,
                        uint channels,uint height,uint width,uint elements,uint pass) {
  uint i=get_global_id(0);if(i>=elements)return;
  uint x=i%width,y=(i/width)%height,c=(i/(width*height))%channels,b=i/(width*height*channels);
  ulong base=((ulong)b*4*channels+pass*channels+c)*height*width+(ulong)y*width;
  ulong src=offset+(ulong)i-x;
  float m=-INFINITY;
  for(int k=-2;k<=2;++k)if((int)x+k>=0 && (int)x+k<(int)width)
   m=pool_max(m,pass?out[base+x+k]:input[src+x+k]);
  temp[i]=m;
  if(!pass)out[base+x]=input[offset+i];
 }
 __kernel void sppf_col(__global const float *temp,__global float *out,
                        uint channels,uint height,uint width,uint elements,uint pass) {
  uint i=get_global_id(0);if(i>=elements)return;
  uint x=i%width,y=(i/width)%height,c=(i/(width*height))%channels,b=i/(width*height*channels);
  ulong src=(ulong)i-y*width;float m=-INFINITY;
  for(int k=-2;k<=2;++k)if((int)y+k>=0 && (int)y+k<(int)height)m=pool_max(m,temp[src+((int)y+k)*width]);
  ulong dst=((ulong)b*4*channels+(pass+1)*channels+c)*height*width+(ulong)y*width+x;
  out[dst]=m;
 }
 )cl";
 cl::Program program(ctx.context(),source);program.build({ctx.device()});programs.emplace(key,program);return program;
}
Tensor sppf_pool_cat(const Tensor &input) {
 GUARD;
 TORCH_CHECK(!at::GradMode::is_enabled(),"SPPF fusion is inference-only");
 TORCH_CHECK(input.device().type()==OpenCLDeviceType && input.scalar_type()==kFloat && input.dim()==4 && input.is_contiguous(),"SPPF expects contiguous FP32 OpenCL NCHW");
 auto b=input.size(0),c=input.size(1),h=input.size(2),w=input.size(3),n=input.numel();
 TORCH_CHECK(n>0 && n<=2147483647 && h>0 && w>0,"Unsupported SPPF shape");
 auto out=new_ocl_tensor({b,4*c,h,w},input.device());auto temp=new_ocl_tensor({b,c,h,w},input.device());
 auto q=getExecutionContext(input);dlprim::Context ctx(q);auto program=sppf_program(ctx);
 cl::Kernel row(program,"sppf_row"),col(program,"sppf_col");
 row.setArg(0,buffer_from_tensor(input));row.setArg(1,cl_ulong(input.storage_offset()));row.setArg(2,buffer_from_tensor(out));row.setArg(3,buffer_from_tensor(temp));
 col.setArg(0,buffer_from_tensor(temp));col.setArg(1,buffer_from_tensor(out));
 for(int j=0;j<4;++j){cl_uint v=j==0?c:j==1?h:j==2?w:n;row.setArg(4+j,v);col.setArg(2+j,v);}
 for(cl_uint pass=0;pass<3;++pass){
  row.setArg(8,pass);col.setArg(6,pass);
  q.queue().enqueueNDRangeKernel(row,cl::NullRange,cl::NDRange((n+255)/256*256),cl::NDRange(256),q.events(),q.event("sppf_row"));
  q.queue().enqueueNDRangeKernel(col,cl::NullRange,cl::NDRange((n+255)/256*256),cl::NDRange(256),q.events(),q.event("sppf_col"));
 }
 sync_if_needed(input.device());return out;
}
}
TORCH_LIBRARY_FRAGMENT(pytorch_ocl,m){m.def("sppf_pool_cat(Tensor input) -> Tensor");}
TORCH_LIBRARY_IMPL(pytorch_ocl,PrivateUse1,m){m.impl("sppf_pool_cat",&ptdlprim::sppf_pool_cat);}
