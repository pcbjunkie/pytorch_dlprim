#include "utils.h"
#include <map>
#include <mutex>
#include <tuple>
#include <limits>
#include <cmath>
#include <algorithm>
namespace ptdlprim {
using namespace at;
static cl::Program detection_program(dlprim::Context &ctx)
{
 static std::mutex mutex;
 static std::map<std::pair<cl_context,cl_device_id>,cl::Program> programs;
 std::lock_guard<std::mutex> lock(mutex);
 auto key=std::make_pair(ctx.context()(),ctx.device()());auto found=programs.find(key);
 if(found!=programs.end())return found->second;
 const char *source=R"cl(
 __kernel void candidates(__global const float *x,ulong xo,uint anchors,uint classes,uint capacity,float threshold,
                          __global float *scores,__global uint *ids,__global uint *counts) {
  uint a=get_global_id(0),b=get_global_id(1);if(a>=anchors)return;
  ulong base=xo+(ulong)b*(classes+4)*anchors+a;
  for(uint c=0;c<4;++c)if(!isfinite(x[base+(ulong)c*anchors]))atomic_or(counts+2*b+1,1u);
  for(uint c=0;c<classes;++c) {
   float score=x[base+(ulong)(c+4)*anchors];
   if(!isfinite(score)){atomic_or(counts+2*b+1,1u);continue;}
   if(score<threshold)continue;
   uint pos=atomic_inc(counts+2*b);
   if(pos<capacity){scores[(ulong)b*capacity+pos]=score;ids[(ulong)b*capacity+pos]=a*classes+c;}
  }
 }
 __kernel void filter_v8(__global const float *x,ulong xo,uint anchors,uint classes,float threshold,
                         __global float *out,__global uint *counts) {
  uint a=get_global_id(0),b=get_global_id(1);if(a>=anchors)return;
  ulong base=xo+(ulong)b*(classes+4)*anchors+a;
  for(uint c=0;c<4;++c)if(!isfinite(x[base+(ulong)c*anchors]))atomic_or(counts+2*b+1,1u);
  float best=-INFINITY;uint label=0;
  for(uint c=0;c<classes;++c) {
   float score=x[base+(ulong)(c+4)*anchors];
   if(!isfinite(score)){atomic_or(counts+2*b+1,1u);continue;}
   if(score>best){best=score;label=c;}
  }
  if(!(best>threshold))return;
  uint pos=atomic_inc(counts+2*b);
  ulong dst=((ulong)b*anchors+pos)*7;
  for(uint c=0;c<4;++c)out[dst+c]=x[base+(ulong)c*anchors];
  out[dst+4]=best;out[dst+5]=(float)label;out[dst+6]=(float)a;
 }
 __kernel void filter_yolox(__global const float *x,ulong xo,uint anchors,uint classes,float threshold,
                            __global float *out,__global uint *counts) {
  uint a=get_global_id(0),b=get_global_id(1);if(a>=anchors)return;
  ulong base=xo+((ulong)b*anchors+a)*(classes+5);
  for(uint c=0;c<5;++c)if(!isfinite(x[base+c]))atomic_or(counts+2*b+1,1u);
  float best=-INFINITY;uint label=0;
  for(uint c=0;c<classes;++c){float v=x[base+5+c];
   if(!isfinite(v)){atomic_or(counts+2*b+1,1u);continue;}
   if(v>best){best=v;label=c;}
  }
  float obj=x[base+4],score=obj*best;
  if(!isfinite(score)){atomic_or(counts+2*b+1,1u);return;}
  if(!(score>=threshold))return;
  uint pos=atomic_inc(counts+2*b);ulong dst=((ulong)b*anchors+pos)*8;
  for(uint c=0;c<5;++c)out[dst+c]=x[base+c];
  out[dst+5]=best;out[dst+6]=(float)label;out[dst+7]=(float)a;
 }
 __kernel void rank_candidates(__global const float *x,ulong xo,uint anchors,uint classes,uint capacity,uint maxdet,
                               __global const float *scores,__global const uint *ids,__global const uint *counts,
                               __global float *out,ulong oo) {
  uint i=get_global_id(0),b=get_global_id(1),n=counts[2*b];
  if(i>=n || n>capacity || counts[2*b+1])return;
  ulong base=(ulong)b*capacity;float score=scores[base+i];uint id=ids[base+i],rank=0;
  for(uint j=0;j<n;++j) {
   float other=scores[base+j];uint otherid=ids[base+j];
   rank+=(other>score || (other==score && otherid<id));
   if(rank>=maxdet)return;
  }
  uint anchor=id/classes,label=id%classes;
  ulong src=xo+(ulong)b*(classes+4)*anchors+anchor;
  ulong dst=oo+((ulong)b*maxdet+rank)*6;
  for(uint c=0;c<4;++c)out[dst+c]=x[src+(ulong)c*anchors];
  out[dst+4]=score;out[dst+5]=(float)label;
 }
 )cl";
 cl::Program program(ctx.context(),source);program.build({ctx.device()});programs.emplace(key,program);return program;
}
std::tuple<Tensor,Tensor> gpu_detections(const Tensor &predictions,double threshold,int64_t maxdet,int64_t capacity)
{
 GUARD;
 TORCH_CHECK(!at::GradMode::is_enabled(),"GPU detector postprocessing is inference-only");
 TORCH_CHECK(predictions.device().type()==OpenCLDeviceType && predictions.scalar_type()==kFloat &&
             predictions.dim()==3 && predictions.is_contiguous(),"Expected contiguous FP32 OpenCL [batch,4+classes,anchors]");
 auto b=predictions.size(0),cols=predictions.size(1),anchors=predictions.size(2);
 TORCH_CHECK(cols>4 && cols<=260 && anchors>=0 && uint64_t(anchors)*uint64_t(cols-4)<=std::numeric_limits<cl_uint>::max(),"Invalid detector shape");
 TORCH_CHECK(std::isfinite(threshold) && threshold>=0 && threshold<=1 && maxdet>0 && maxdet<=300 &&
             capacity>=maxdet && capacity<=4096,"Invalid detector threshold/top-k/candidate budget");
 maxdet=std::min(maxdet,anchors);
 auto out=new_ocl_tensor({b,maxdet,6},predictions.device());out.zero_();
 // counts[:,0] is the untruncated eligible count; counts[:,1] flags nonfinite input.
 auto counts=new_ocl_tensor({b,2},predictions.device(),kInt);counts.zero_();
 if(!b || !anchors)return {out,counts};
 auto scores=new_ocl_tensor({b,capacity},predictions.device());
 auto ids=new_ocl_tensor({b,capacity},predictions.device(),kInt);
 auto q=getExecutionContext(predictions);dlprim::Context ctx(q);auto program=detection_program(ctx);
 cl::Kernel filter(program,"candidates"),rank(program,"rank_candidates");int p=0;
 filter.setArg(p++,buffer_from_tensor(predictions));filter.setArg(p++,cl_ulong(predictions.storage_offset()));
 filter.setArg(p++,cl_uint(anchors));filter.setArg(p++,cl_uint(cols-4));filter.setArg(p++,cl_uint(capacity));filter.setArg(p++,cl_float(threshold));
 filter.setArg(p++,buffer_from_tensor(scores));filter.setArg(p++,buffer_from_tensor(ids));filter.setArg(p++,buffer_from_tensor(counts));
 q.queue().enqueueNDRangeKernel(filter,cl::NullRange,cl::NDRange((anchors+127)/128*128,b),cl::NDRange(128,1),q.events(),q.event("detector_candidates"));
 p=0;rank.setArg(p++,buffer_from_tensor(predictions));rank.setArg(p++,cl_ulong(predictions.storage_offset()));
 rank.setArg(p++,cl_uint(anchors));rank.setArg(p++,cl_uint(cols-4));rank.setArg(p++,cl_uint(capacity));rank.setArg(p++,cl_uint(maxdet));
 rank.setArg(p++,buffer_from_tensor(scores));rank.setArg(p++,buffer_from_tensor(ids));rank.setArg(p++,buffer_from_tensor(counts));
 rank.setArg(p++,buffer_from_tensor(out));rank.setArg(p++,cl_ulong(out.storage_offset()));
 q.queue().enqueueNDRangeKernel(rank,cl::NullRange,cl::NDRange((capacity+127)/128*128,b),cl::NDRange(128,1),q.events(),q.event("detector_rank_gather"));
 sync_if_needed(predictions.device());return {out,counts};
}
std::tuple<Tensor,Tensor> gpu_v8_candidates(const Tensor &predictions,double threshold)
{
 GUARD;
 TORCH_CHECK(!at::GradMode::is_enabled(),"GPU detector postprocessing is inference-only");
 TORCH_CHECK(predictions.device().type()==OpenCLDeviceType && predictions.scalar_type()==kFloat &&
             predictions.dim()==3 && predictions.is_contiguous(),"Expected contiguous FP32 OpenCL [batch,4+classes,anchors]");
 auto b=predictions.size(0),cols=predictions.size(1),anchors=predictions.size(2);
 TORCH_CHECK(cols>4 && cols<=260 && anchors>=0 && anchors<=16777216,"Invalid detector shape");
 TORCH_CHECK(std::isfinite(threshold) && threshold>=0 && threshold<=1,"Invalid detector threshold");
 // Full anchor capacity: no premature top-k or candidate truncation before NMS.
 auto out=new_ocl_tensor({b,anchors,7},predictions.device());out.zero_();
 auto counts=new_ocl_tensor({b,2},predictions.device(),kInt);counts.zero_();
 if(!b || !anchors)return {out,counts};
 auto q=getExecutionContext(predictions);dlprim::Context ctx(q);auto program=detection_program(ctx);
 cl::Kernel filter(program,"filter_v8");int p=0;
 filter.setArg(p++,buffer_from_tensor(predictions));filter.setArg(p++,cl_ulong(predictions.storage_offset()));
 filter.setArg(p++,cl_uint(anchors));filter.setArg(p++,cl_uint(cols-4));filter.setArg(p++,cl_float(threshold));
 filter.setArg(p++,buffer_from_tensor(out));filter.setArg(p++,buffer_from_tensor(counts));
 q.queue().enqueueNDRangeKernel(filter,cl::NullRange,cl::NDRange((anchors+127)/128*128,b),cl::NDRange(128,1),q.events(),q.event("detector_v8_candidates"));
 sync_if_needed(predictions.device());return {out,counts};
}
std::tuple<Tensor,Tensor> gpu_yolox_candidates(const Tensor &predictions,double threshold)
{
 GUARD;
 TORCH_CHECK(!at::GradMode::is_enabled(),"GPU detector postprocessing is inference-only");
 TORCH_CHECK(predictions.device().type()==OpenCLDeviceType && predictions.scalar_type()==kFloat &&
             predictions.dim()==3 && predictions.is_contiguous(),"Expected contiguous FP32 OpenCL [batch,anchors,5+classes]");
 auto b=predictions.size(0),anchors=predictions.size(1),cols=predictions.size(2);
 TORCH_CHECK(cols>5 && cols<=261 && anchors>=0 && anchors<=16777216,"Invalid detector shape");
 TORCH_CHECK(std::isfinite(threshold) && threshold>=0 && threshold<=1,"Invalid detector threshold");
 auto out=new_ocl_tensor({b,anchors,8},predictions.device());out.zero_();
 auto counts=new_ocl_tensor({b,2},predictions.device(),kInt);counts.zero_();
 if(!b || !anchors)return {out,counts};
 auto q=getExecutionContext(predictions);dlprim::Context ctx(q);auto program=detection_program(ctx);
 cl::Kernel filter(program,"filter_yolox");int p=0;
 filter.setArg(p++,buffer_from_tensor(predictions));filter.setArg(p++,cl_ulong(predictions.storage_offset()));
 filter.setArg(p++,cl_uint(anchors));filter.setArg(p++,cl_uint(cols-5));filter.setArg(p++,cl_float(threshold));
 filter.setArg(p++,buffer_from_tensor(out));filter.setArg(p++,buffer_from_tensor(counts));
 q.queue().enqueueNDRangeKernel(filter,cl::NullRange,cl::NDRange((anchors+127)/128*128,b),cl::NDRange(128,1),q.events(),q.event("detector_yolox_candidates"));
 sync_if_needed(predictions.device());return {out,counts};
}

}
TORCH_LIBRARY_FRAGMENT(pytorch_ocl,m) {
 m.def("gpu_yolox_candidates(Tensor predictions, float threshold) -> (Tensor, Tensor)");
 m.def("gpu_v8_candidates(Tensor predictions, float threshold) -> (Tensor, Tensor)");
 m.def("gpu_detections(Tensor predictions, float threshold, int maxdet, int capacity) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(pytorch_ocl,PrivateUse1,m) {
 m.impl("gpu_yolox_candidates",&ptdlprim::gpu_yolox_candidates);
 m.impl("gpu_v8_candidates",&ptdlprim::gpu_v8_candidates);
 m.impl("gpu_detections",&ptdlprim::gpu_detections);
}
