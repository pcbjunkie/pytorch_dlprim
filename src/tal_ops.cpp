#include "utils.h"
#include <map>
#include <mutex>
#include <tuple>
#include <cmath>
namespace ptdlprim {
using namespace at;
static cl::Program tal_program(dlprim::Context &ctx,bool with_targets=false) {
 static std::mutex mutex;
 static std::map<std::tuple<cl_context,cl_device_id,bool>,cl::Program> cache;
 std::lock_guard<std::mutex> lock(mutex);
 auto key=std::make_tuple(ctx.context()(),ctx.device()(),with_targets);auto it=cache.find(key);
 if(it!=cache.end())return it->second;
 const char *source=R"cl(
#pragma OPENCL FP_CONTRACT OFF
inline float tal_min(float a,float b){return isnan(a) || isnan(b) ? NAN : fmin(a,b);}
inline float tal_max(float a,float b){return isnan(a) || isnan(b) ? NAN : fmax(a,b);}
__kernel void tal_metrics(__global const float*s,ulong so,__global const float*p,ulong po,
 __global const float*l,ulong lo,__global const float*g,ulong go,__global const uchar*m,ulong mo,
 __global float*align,__global float*over,__global int*err,uint A,uint G,uint C,float alpha,float beta) {
 uint a=get_global_id(0),bg=get_global_id(1);if(a>=A)return;
 uint b=bg/G;float label=l[lo+bg];int cls=isfinite(label) ? convert_int_sat_rtz(label) : -1;
 if(cls<0 && isfinite(label))cls+=(int)C;
 if(cls<0 || cls>=C){if(a==0)atomic_or(err,1);cls=0;}
 ulong q=(ulong)bg*A+a;
 if(!m[mo+q]){align[q]=0;over[q]=0;return;}
 ulong i=po+((ulong)b*A+a)*4,j=go+(ulong)bg*4;
 float x1=g[j],y1=g[j+1],x2=g[j+2],y2=g[j+3];
 float X1=p[i],Y1=p[i+1],X2=p[i+2],Y2=p[i+3];
 float w1=x2-x1,h1=y2-y1+1e-7f,w2=X2-X1,h2=Y2-Y1+1e-7f;
 float inter=tal_max(tal_min(x2,X2)-tal_max(x1,X1),0.0f)*tal_max(tal_min(y2,Y2)-tal_max(y1,Y1),0.0f);
 float u=w1*h1+w2*h2-inter+1e-7f,iou=inter/u;
 float cw=tal_max(x2,X2)-tal_min(x1,X1),ch=tal_max(y2,Y2)-tal_min(y1,Y1);
 float c2=cw*cw+ch*ch+1e-7f,rx=X1+X2-x1-x2,ry=Y1+Y2-y1-y2;
 float rho=(rx*rx+ry*ry)/4.0f,d=atan(w2/h2)-atan(w1/h1);
 float v=0.4052847345693511f*d*d,al=v/(v-iou+(1.0f+1e-7f));
 float ov=tal_max(iou-(rho/c2+v*al),0.0f);over[q]=ov;
 float score=s[so+((ulong)b*A+a)*C+cls];
 align[q]=pow(score,alpha)*pow(ov,beta);
}
// Recreate the native scatter-add mask including its duplicate-index rule.
__kernel void tal_topk_mask(__global const long *indices, ulong io,
                            __global const uchar *valid, ulong vo,
                            __global char *output, uint anchors, uint k)
{
    uint a=get_global_id(0),row=get_global_id(1);
    if(a>=anchors)return;
    ulong base=(ulong)row*k;
    uint count=0;
    for(uint j=0;j<k;j++) {
        long selected=valid[vo+base+j] ? indices[io+base+j] : 0;
        count+=(selected==a);
    }
    output[(ulong)row*anchors+a]=(count==1);
}
#ifdef TAL_TARGETS
__kernel void tal_targets(
 __global const float *labels,ulong lo,__global const float *boxes,ulong bo,
 __global const long *indices,ulong io,__global const uchar *foreground,ulong fo,
 __global long *out_labels,__global float *out_boxes,__global char *out_scores,
 __global int *error,uint anchors,uint ground_truths,uint classes,uint batch)
{
 ulong item=get_global_id(0);
 if(item>=(ulong)anchors*batch)return;
 long index=indices[io+item];
 if(index<0 || index>=ground_truths) {
  atomic_or(error,1);
  index=0;
 }
 ulong target=(item/anchors)*ground_truths+index;
 float label=labels[lo+target];
 long cls=isfinite(label) ? max((long)0,convert_long_sat_rtz(label)) : (long)-1;
 if(cls<0 || cls>=classes) {
  atomic_or(error,2);
  cls=0;
 }
 out_labels[item]=cls;
 vstore4(vload4(target,boxes+bo),item,out_boxes);
 for(uint c=0;c<classes;++c)
  out_scores[item*classes+c]=(char)(foreground[fo+item] && c==cls);
}
#endif
)cl";
 cl::Program program(ctx.context(),source);
 try {program.build({ctx.device()},with_targets ? "-DTAL_TARGETS=1" : "");}catch(...){std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device());throw;}
 return cache.emplace(key,program).first->second;
}
std::tuple<Tensor,Tensor> tal_box_metrics(const Tensor &scores,const Tensor &boxes,const Tensor &labels,const Tensor &gt,const Tensor &mask,double alpha,double beta) {
 GUARD;
 TORCH_CHECK(!at::GradMode::is_enabled(),"TAL metrics are assignment-only, no gradients");
 TORCH_CHECK(scores.device().type()==OpenCLDeviceType && scores.scalar_type()==kFloat && scores.dim()==3,"Expected FP32 OpenCL [B,A,C] scores");
 auto B=scores.size(0),A=scores.size(1),C=scores.size(2);
 TORCH_CHECK(boxes.sizes()==IntArrayRef({B,A,4}) && gt.dim()==3 && gt.size(0)==B && gt.size(2)==4,"Invalid TAL box shapes");
 auto G=gt.size(1);
 TORCH_CHECK(labels.sizes()==IntArrayRef({B,G,1}) && mask.sizes()==IntArrayRef({B,G,A}),"Invalid TAL labels/mask shapes");
 for(auto *v:{&boxes,&labels,&gt})TORCH_CHECK(v->device()==scores.device() && v->scalar_type()==kFloat,"TAL inputs must share FP32 device");
 TORCH_CHECK(mask.device()==scores.device() && mask.scalar_type()==kBool,"TAL mask must be boolean on scores device");
 TORCH_CHECK(B>=0 && G>=0 && A>=0 && C>0 && B*G<=2147483647 && A<=2147483647 && C<=2147483647 && std::isfinite(alpha) && std::isfinite(beta) && alpha>0 && beta>0,"Invalid TAL dimensions/exponents");
 auto s=scores.contiguous(),p=boxes.contiguous(),l=labels.contiguous(),g=gt.contiguous(),m=mask.contiguous();
 auto align=at::empty({B,G,A},scores.options()),over=at::empty_like(align);
 if(!B || !G || !A)return {align,over};
 auto err=at::zeros({1},scores.options().dtype(kInt));auto q=getExecutionContext(scores);dlprim::Context ctx(q);auto prog=tal_program(ctx);cl::Kernel kernel(prog,"tal_metrics");int arg=0;
 for(auto *v:{&s,&p,&l,&g,&m}){kernel.setArg(arg++,buffer_from_tensor(*v));kernel.setArg(arg++,cl_ulong(v->storage_offset()));}
 kernel.setArg(arg++,buffer_from_tensor(align));kernel.setArg(arg++,buffer_from_tensor(over));kernel.setArg(arg++,buffer_from_tensor(err));
 kernel.setArg(arg++,cl_uint(A));kernel.setArg(arg++,cl_uint(G));kernel.setArg(arg++,cl_uint(C));kernel.setArg(arg++,cl_float(alpha));kernel.setArg(arg++,cl_float(beta));
 q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange((A+63)/64*64,B*G),cl::NDRange(64,1),q.events(),q.event("tal_box_metrics"));
 sync_if_needed(scores.device());TORCH_CHECK(err.item<int>()==0,"TAL class index out of bounds");return {align,over};
}
std::tuple<Tensor,Tensor,Tensor> tal_targets(const Tensor &labels,const Tensor &boxes,const Tensor &indices,
                                            const Tensor &foreground,int64_t classes)
{
 GUARD;
 TORCH_CHECK(!at::GradMode::is_enabled(),"TAL targets are assignment-only");
 TORCH_CHECK(labels.device().type()==OpenCLDeviceType && labels.scalar_type()==kFloat &&
             labels.dim()==3 && labels.size(2)==1,"Expected FP32 OpenCL [B,G,1] labels");
 const int64_t batch=labels.size(0),ground_truths=labels.size(1);
 TORCH_CHECK(indices.dim()==2 && indices.size(0)==batch && indices.scalar_type()==kLong,
             "Expected int64 [B,A] indices");
 const int64_t anchors=indices.size(1);
 TORCH_CHECK(boxes.sizes()==IntArrayRef({batch,ground_truths,4}) && boxes.scalar_type()==kFloat &&
             foreground.sizes()==indices.sizes() && foreground.scalar_type()==kBool,
             "Invalid TAL target inputs");
 for(auto *input:{&boxes,&indices,&foreground})
  TORCH_CHECK(input->device()==labels.device(),"TAL targets require a shared device");
 TORCH_CHECK(batch>0 && ground_truths>0 && anchors>0 &&
             batch<=65535 && ground_truths<=65535 && anchors<=2147483647 && classes>=1 && classes<=128,
             "TAL target dimensions out of range");
 auto q=getExecutionContext(labels);
 dlprim::Context ctx(q);
 TORCH_CHECK(ctx.device().getInfo<CL_DEVICE_ADDRESS_BITS>()==64,"TAL target kernel requires a 64-bit device");
 auto l=labels.contiguous(),b=boxes.contiguous(),idx=indices.contiguous(),fg=foreground.contiguous();
 auto out_labels=at::empty_like(idx);
 auto out_boxes=at::empty({batch,anchors,4},labels.options());
 auto out_scores=at::empty({batch,anchors,classes},labels.options().dtype(kChar));
 auto status=at::zeros({1},labels.options().dtype(kInt));
 cl::Kernel kernel(tal_program(ctx,true),"tal_targets");
 int arg=0;
 for(auto *input:{&l,&b,&idx,&fg}) {
  kernel.setArg(arg++,buffer_from_tensor(*input));
  kernel.setArg(arg++,cl_ulong(input->storage_offset()));
 }
 for(auto *output:{&out_labels,&out_boxes,&out_scores,&status})
  kernel.setArg(arg++,buffer_from_tensor(*output));
 for(auto size:{anchors,ground_truths,classes,batch})
  kernel.setArg(arg++,cl_uint(size));
 const size_t items=size_t(batch)*size_t(anchors);
 q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange((items+63)/64*64),
                               cl::NDRange(64),q.events(),q.event("tal_targets"));
 sync_if_needed(labels.device());
 int error=status.item<int>();
 TORCH_CHECK(!(error&1),"TAL target index out of bounds");
 TORCH_CHECK(!(error&2),"TAL target class out of bounds");
 return {out_labels,out_boxes,out_scores};
}

Tensor tal_topk_mask(const Tensor &indices,const Tensor &valid,int64_t anchors)
{
    GUARD;
    TORCH_CHECK(indices.device().type()==OpenCLDeviceType && indices.scalar_type()==kLong &&
                indices.dim()==3 && indices.size(2)>0 && indices.size(2)<=32 &&
                valid.device()==indices.device() && valid.scalar_type()==kBool &&
                valid.sizes()==indices.sizes() && anchors>0 && anchors<=2147483647,
                "Expected OpenCL int64 [B,G,K] top-k indices, matching validity and an anchor count");
    const auto batch=indices.size(0),ground_truths=indices.size(1),k=indices.size(2);
    TORCH_CHECK(batch*ground_truths<=2147483647,"TAL top-k row count too large");
    auto output=at::empty({batch,ground_truths,anchors},indices.options().dtype(kChar));
    if(!output.numel())return output;
    auto idx=indices.contiguous(),mask=valid.contiguous();
    auto q=getExecutionContext(indices);
    dlprim::Context ctx(q);
    cl::Kernel kernel(tal_program(ctx),"tal_topk_mask");
    int arg=0;
    for(auto *input:{&idx,&mask}) {
        kernel.setArg(arg++,buffer_from_tensor(*input));
        kernel.setArg(arg++,cl_ulong(input->storage_offset()));
    }
    kernel.setArg(arg++,buffer_from_tensor(output));
    kernel.setArg(arg++,cl_uint(anchors));
    kernel.setArg(arg++,cl_uint(k));
    q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,
        cl::NDRange((anchors+63)/64*64,batch*ground_truths),cl::NDRange(64,1),
        q.events(),q.event("tal_topk_mask"));
    sync_if_needed(indices.device());
    return output;
}

}
TORCH_LIBRARY_FRAGMENT(pytorch_ocl,m){
 m.def("tal_box_metrics(Tensor scores, Tensor boxes, Tensor labels, Tensor gt, Tensor mask, float alpha, float beta) -> (Tensor, Tensor)");
 m.def("tal_targets(Tensor labels, Tensor boxes, Tensor indices, Tensor foreground, int classes) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(pytorch_ocl,PrivateUse1,m){m.impl("tal_box_metrics",&ptdlprim::tal_box_metrics);m.impl("tal_targets",&ptdlprim::tal_targets);}
// no_grad retains the AutogradPrivateUse1 key; the kernel explicitly rejects
// enabled autograd because assignment metrics are detached by definition.
TORCH_LIBRARY_IMPL(pytorch_ocl,AutogradPrivateUse1,m){m.impl("tal_box_metrics",&ptdlprim::tal_box_metrics);m.impl("tal_targets",&ptdlprim::tal_targets);}

TORCH_LIBRARY_FRAGMENT(pytorch_ocl,m) {
    m.def("tal_topk_mask(Tensor indices, Tensor valid, int anchors) -> Tensor");
}
TORCH_LIBRARY_IMPL(pytorch_ocl,PrivateUse1,m) {m.impl("tal_topk_mask",&ptdlprim::tal_topk_mask);}
TORCH_LIBRARY_IMPL(pytorch_ocl,AutogradPrivateUse1,m) {m.impl("tal_topk_mask",&ptdlprim::tal_topk_mask);}
