#include "utils.h"
#include <map>
#include <mutex>
namespace ptdlprim {
using namespace at;
static cl::Program focus_program(dlprim::Context &ctx){
 static std::mutex mutex;static std::map<std::pair<cl_context,cl_device_id>,cl::Program> programs;
 std::lock_guard<std::mutex> lock(mutex);auto key=std::make_pair(ctx.context()(),ctx.device()());auto i=programs.find(key);if(i!=programs.end())return i->second;
 const char *source=R"cl(
 __kernel void focus_bgr(__global const uchar *x,ulong offset,uint height,uint width,uint oh,uint ow,__global float *out,float padding){
  uint p=get_global_id(0),bc=get_global_id(1);if(p>=oh*ow)return;
  uint b=bc/12,c=bc%12,block=c/3,channel=c%3;
  uint y=(p/ow)*2+block%2,z=(p%ow)*2+block/2;
  float value=padding;
  if(y<height && z<width)value=(float)x[offset+(((ulong)b*height+y)*width+z)*3+channel];
  out[(ulong)bc*oh*ow+p]=value;
 }
 )cl";
 cl::Program program(ctx.context(),source);program.build({ctx.device()});programs.emplace(key,program);return program;
}
Tensor focus_bgr_u8(const Tensor &input){
 GUARD;
 TORCH_CHECK(!at::GradMode::is_enabled(),"Fused input Focus is inference-only");
 TORCH_CHECK(input.device().type()==OpenCLDeviceType && input.scalar_type()==kByte && input.dim()==4 && input.size(3)==3 && input.is_contiguous(),"Expected contiguous OpenCL uint8 BGR [batch,height,width,3]");
 auto b=input.size(0),h=input.size(1),w=input.size(2);TORCH_CHECK(h>0 && w>0 && h*w<2147483647,"Invalid input shape");
 auto oh=(h+31)/32*16,ow=(w+31)/32*16;
 auto out=new_ocl_tensor({b,12,oh,ow},input.device());if(!b)return out;
 auto q=getExecutionContext(input);dlprim::Context ctx(q);auto program=focus_program(ctx);cl::Kernel kernel(program,"focus_bgr");
 kernel.setArg(0,buffer_from_tensor(input));kernel.setArg(1,cl_ulong(input.storage_offset()));kernel.setArg(2,cl_uint(h));kernel.setArg(3,cl_uint(w));kernel.setArg(4,cl_uint(oh));kernel.setArg(5,cl_uint(ow));kernel.setArg(6,buffer_from_tensor(out));kernel.setArg(7,cl_float(114.0f));
 q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange((oh*ow+255)/256*256,b*12),cl::NDRange(256,1),q.events(),q.event("focus_bgr_u8"));
 sync_if_needed(input.device());return out;
}
}
TORCH_LIBRARY_FRAGMENT(pytorch_ocl,m){m.def("focus_bgr_u8(Tensor input) -> Tensor");}
TORCH_LIBRARY_IMPL(pytorch_ocl,PrivateUse1,m){m.impl("focus_bgr_u8",&ptdlprim::focus_bgr_u8);}
