#include "utils.h"
#include <array>
#include <map>
#include <mutex>
#include <tuple>
namespace ptdlprim {
using namespace at;
static cl::Program bbox_loss_program(dlprim::Context &ctx) {
 static std::mutex mutex;static std::map<std::pair<cl_context,cl_device_id>,cl::Program> cache;
 std::lock_guard<std::mutex> lock(mutex);auto key=std::make_pair(ctx.context()(),ctx.device()());auto it=cache.find(key);if(it!=cache.end())return it->second;
 const char *source=R"cl(
#pragma OPENCL FP_CONTRACT OFF
// Four-component forward derivatives with respect to predicted xyxy.
typedef struct {float v;float4 g;} D;
inline D dc(float v){D d={v,(float4)(0)};return d;}
inline D dv(float v,float4 g){D d={v,g};return d;}
inline D da(D a,D b){return dv(a.v+b.v,a.g+b.g);}
inline D ds(D a,D b){return dv(a.v-b.v,a.g-b.g);}
inline D dm(D a,D b){return dv(a.v*b.v,a.g*b.v+b.g*a.v);}
inline D dd(D a,D b){return dv(a.v/b.v,(a.g*b.v-b.g*a.v)/(b.v*b.v));}
inline D datan(D a){return dv(atan(a.v),a.g/(1+a.v*a.v));}
inline D dmin(D a,D b){if(isnan(a.v)||isnan(b.v))return dv(NAN,(float4)(NAN));if(a.v<b.v)return a;if(a.v>b.v)return b;return dv(a.v,(a.g+b.g)*.5f);}
inline D dmax(D a,D b){if(isnan(a.v)||isnan(b.v))return dv(NAN,(float4)(NAN));if(a.v>b.v)return a;if(a.v<b.v)return b;return dv(a.v,(a.g+b.g)*.5f);}
inline D dclamp0(D a){return a.v<0 ? dc(0) : a;}
inline D box_ciou(__global const float *p, ulong q, __global const float *t, ulong gt) {
 D x1=dv(p[q],(float4)(1,0,0,0)),y1=dv(p[q+1],(float4)(0,1,0,0)),x2=dv(p[q+2],(float4)(0,0,1,0)),y2=dv(p[q+3],(float4)(0,0,0,1));
 D X1=dc(t[gt]),Y1=dc(t[gt+1]),X2=dc(t[gt+2]),Y2=dc(t[gt+3]);
 D w1=ds(x2,x1),h1=da(ds(y2,y1),dc(1e-7f)),w2=ds(X2,X1),h2=da(ds(Y2,Y1),dc(1e-7f));
 D inter=dm(dclamp0(ds(dmin(x2,X2),dmax(x1,X1))),dclamp0(ds(dmin(y2,Y2),dmax(y1,Y1))));
 D un=da(ds(da(dm(w1,h1),dm(w2,h2)),inter),dc(1e-7f)),iou=dd(inter,un);
 D cw=ds(dmax(x2,X2),dmin(x1,X1)),ch=ds(dmax(y2,Y2),dmin(y1,Y1));D c2=da(da(dm(cw,cw),dm(ch,ch)),dc(1e-7f));
 D rx=ds(ds(da(X1,X2),x1),x2),ry=ds(ds(da(Y1,Y2),y1),y2),rho=dm(da(dm(rx,rx),dm(ry,ry)),dc(.25f));
 D delta=ds(datan(dd(w2,h2)),datan(dd(w1,h1))),v=dm(dc(.4052847345693511f),dm(delta,delta));
 // Ultralytics deliberately detaches alpha; its derivative must remain zero.
 float alpha=v.v/(v.v-iou.v+(1.0f+1e-7f));return ds(iou,da(dd(rho,c2),dm(v,dc(alpha))));
}
__kernel void bbox_loss16(__global const float*d,ulong doff,__global const float*p,ulong poff,
 __global const float*ap,ulong ao,__global const float*t,ulong toff,__global const float*s,ulong so,
 __global const float*norm,ulong no,__global const uchar*mask,ulong mo,
 __global float*li,__global float*ld,__global float*gd,__global float*gp,uint A,uint C) {
 uint a=get_global_id(0),b=get_global_id(1);if(a>=A)return;ulong i=(ulong)b*A+a;
 li[i]=0;ld[i]=0;if(!mask[mo+i])return;
 float weight=0;for(uint c=0;c<C;c++)weight+=s[so+i*C+c];float scale=weight/norm[no];
 ulong q=poff+i*4,gt=toff+i*4;
 D ciou=box_ciou(p,q,t,gt);
 li[i]=(1-ciou.v)*scale;float4 grad=-ciou.g*scale;vstore4(grad,0,gp+i*4);
 float targets[4]={ap[ao+a*2]-t[gt],ap[ao+a*2+1]-t[gt+1],t[gt+2]-ap[ao+a*2],t[gt+3]-ap[ao+a*2+1]};float loss=0;
 for(uint edge=0;edge<4;edge++) {
  float target=clamp(targets[edge],0.0f,14.99f);uint left=(uint)target,right=left+1;float wl=(float)right-target,wr=1-wl;
  ulong base=doff+i*64+edge*16;float maximum=-INFINITY;
  for(uint k=0;k<16;k++)maximum=fmax(maximum,d[base+k]);
  float ex[16],sum=0;
  for(uint k=0;k<16;k++){ex[k]=exp(d[base+k]-maximum);sum+=ex[k];}
  float logsum=log(sum);loss+=((logsum-(d[base+left]-maximum))*wl+(logsum-(d[base+right]-maximum))*wr)*.25f;
  for(uint k=0;k<16;k++)gd[i*64+edge*16+k]=(ex[k]/sum-(k==left ? wl : 0)-(k==right ? wr : 0))*(scale*.25f);
 }
 ld[i]=loss*scale;
}

__kernel void bbox_loss_l1(
    __global const float *dist, ulong d_offset, __global const float *boxes, ulong b_offset,
    __global const float *anchors, ulong a_offset, __global const float *target, ulong t_offset,
    __global const float *scores, ulong s_offset, __global const float *norm, ulong n_offset,
    __global const uchar *mask, ulong m_offset, __global const float *extent, ulong e_offset,
    __global const float *stride, ulong st_offset, __global float *iou_loss, __global float *l1_loss,
    __global float *dist_grad, __global float *box_grad, uint anchor_count, uint class_count)
{
    uint a=get_global_id(0);
    if(a>=anchor_count)return;
    ulong i=(ulong)get_global_id(1)*anchor_count+a;
    if(!mask[m_offset+i]) {
        iou_loss[i]=0.0f;
        l1_loss[i]=0.0f;
        vstore4((float4)(0),i,dist_grad);
        vstore4((float4)(0),i,box_grad);
        return;
    }

    float weight=0.0f;
    for(uint c=0;c<class_count;c++)
        weight+=scores[s_offset+i*class_count+c];
    float scale=weight/norm[n_offset];
    ulong gt=t_offset+i*4;
    D ciou=box_ciou(boxes,b_offset+i*4,target,gt);
    iou_loss[i]=(1.0f-ciou.v)*scale;
    vstore4(-ciou.g*scale,i,box_grad);

    float2 center=vload2(a,anchors+a_offset);
    float4 bounds=vload4(0,target+gt);
    float4 target_dist=(float4)(center-bounds.xy,bounds.zw-center);
    float4 pred_dist=vload4(i,dist+d_offset);
    float stride_value=stride[st_offset+a];
    float2 hw=vload2(0,extent+e_offset);
    float4 divisor=(float4)(hw.y,hw.x,hw.y,hw.x);
    float4 error=pred_dist*stride_value/divisor-target_dist*stride_value/divisor;
    float4 absolute=fabs(error);
    l1_loss[i]=((absolute.x+absolute.y+absolute.z+absolute.w)*0.25f)*scale;
    float4 grad=((sign(error)*(scale*0.25f))/divisor)*stride_value;
    vstore4(grad,i,dist_grad);
}
)cl";
 cl::Program program(ctx.context(),source);try{program.build({ctx.device()});}catch(...){std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device());throw;}
 return cache.emplace(key,program).first->second;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> bbox_loss16(const Tensor &dist,const Tensor &boxes,const Tensor &anchors,const Tensor &target,const Tensor &scores,const Tensor &normalizer,const Tensor &mask) {
 GUARD;TORCH_CHECK(!at::GradMode::is_enabled(),"bbox_loss16 uses explicit first-order autograd wrapper");
 TORCH_CHECK(dist.device().type()==OpenCLDeviceType && dist.scalar_type()==kFloat && dist.dim()==3 && dist.size(2)==64,"Expected FP32 OpenCL [B,A,64]");
 auto B=dist.size(0),A=dist.size(1);TORCH_CHECK(A<=65535 && scores.dim()==3 && scores.size(0)==B && scores.size(1)==A && scores.size(2)>0 && scores.size(2)<=1024,"Unsupported fused box loss shape");
 TORCH_CHECK(boxes.sizes()==IntArrayRef({B,A,4}) && target.sizes()==boxes.sizes() && anchors.sizes()==IntArrayRef({A,2}) && mask.sizes()==IntArrayRef({B,A}) && normalizer.numel()==1,"Invalid box loss input shapes");
 for(auto *v:{&boxes,&anchors,&target,&scores,&normalizer})TORCH_CHECK(v->device()==dist.device() && v->scalar_type()==kFloat,"Box loss inputs must share FP32 GPU device");
 TORCH_CHECK(mask.device()==dist.device() && mask.scalar_type()==kBool,"Expected boolean GPU foreground mask");
 auto d=dist.contiguous(),p=boxes.contiguous(),a=anchors.contiguous(),t=target.contiguous(),s=scores.contiguous(),n=normalizer.contiguous(),m=mask.contiguous();
 auto gd=at::zeros_like(d),gp=at::zeros_like(p),li=at::empty({B,A},dist.options()),ld=at::empty_like(li);
 if(B && A){auto q=getExecutionContext(dist);dlprim::Context ctx(q);auto program=bbox_loss_program(ctx);cl::Kernel kernel(program,"bbox_loss16");int arg=0;
  for(auto *v:{&d,&p,&a,&t,&s,&n,&m}){kernel.setArg(arg++,buffer_from_tensor(*v));kernel.setArg(arg++,cl_ulong(v->storage_offset()));}
  for(auto *v:{&li,&ld,&gd,&gp}) {kernel.setArg(arg++,buffer_from_tensor(*v));}
  kernel.setArg(arg++,cl_uint(A));kernel.setArg(arg++,cl_uint(scores.size(2)));
  q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange((A+63)/64*64,B),cl::NDRange(64,1),q.events(),q.event("bbox_loss16"));sync_if_needed(dist.device());}
 return {li.sum(),ld.sum(),gd,gp};
}

std::tuple<Tensor,Tensor,Tensor,Tensor> bbox_loss_l1(
    const Tensor &dist,const Tensor &boxes,const Tensor &anchors,const Tensor &target,
    const Tensor &scores,const Tensor &normalizer,const Tensor &mask,const Tensor &extent,
    const Tensor &stride)
{
    GUARD;
    TORCH_CHECK(!at::GradMode::is_enabled(),"bbox_loss_l1 uses explicit first-order autograd");
    TORCH_CHECK(dist.device().type()==OpenCLDeviceType && dist.scalar_type()==kFloat &&
                dist.dim()==3 && dist.size(2)==4,"Expected FP32 OpenCL [B,A,4]");
    const auto b=dist.size(0),a=dist.size(1);
    TORCH_CHECK(a<=65535 && boxes.sizes()==dist.sizes() && target.sizes()==dist.sizes() &&
                anchors.sizes()==IntArrayRef({a,2}) && mask.sizes()==IntArrayRef({b,a}) &&
                scores.dim()==3 && scores.size(0)==b && scores.size(1)==a &&
                scores.size(2)>0 && scores.size(2)<=1024 &&
                normalizer.numel()==1 && extent.sizes()==IntArrayRef({2}) &&
                stride.sizes()==IntArrayRef({a,1}),"Invalid CIoU/L1 input shapes");
    for(auto *tensor:{&boxes,&anchors,&target,&scores,&normalizer,&extent,&stride})
        TORCH_CHECK(tensor->device()==dist.device() && tensor->scalar_type()==kFloat,
                    "CIoU/L1 inputs must share the FP32 OpenCL device");
    TORCH_CHECK(mask.device()==dist.device() && mask.scalar_type()==kBool,
                "Expected boolean OpenCL foreground mask");

    std::array<Tensor,9> inputs={dist.contiguous(),boxes.contiguous(),anchors.contiguous(),
        target.contiguous(),scores.contiguous(),normalizer.contiguous(),mask.contiguous(),
        extent.contiguous(),stride.contiguous()};
    auto li=at::empty({b,a},dist.options()),ll=at::empty_like(li);
    auto gd=at::empty(dist.sizes(),dist.options()),gp=at::empty(boxes.sizes(),boxes.options());
    if(b && a) {
        auto q=getExecutionContext(dist);
        dlprim::Context ctx(q);
        cl::Kernel kernel(bbox_loss_program(ctx),"bbox_loss_l1");
        int arg=0;
        for(auto &tensor:inputs) {
            kernel.setArg(arg++,buffer_from_tensor(tensor));
            kernel.setArg(arg++,cl_ulong(tensor.storage_offset()));
        }
        for(auto *tensor:{&li,&ll,&gd,&gp})
            kernel.setArg(arg++,buffer_from_tensor(*tensor));
        kernel.setArg(arg++,cl_uint(a));
        kernel.setArg(arg++,cl_uint(scores.size(2)));
        q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange((a+63)/64*64,b),
                                       cl::NDRange(64,1),q.events(),q.event("bbox_loss_l1"));
        sync_if_needed(dist.device());
    }
    return {li.sum(),ll.sum(),gd,gp};
}
}
TORCH_LIBRARY_FRAGMENT(pytorch_ocl,m){m.def("bbox_loss16(Tensor dist, Tensor boxes, Tensor anchors, Tensor target, Tensor scores, Tensor normalizer, Tensor mask) -> (Tensor, Tensor, Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(pytorch_ocl,PrivateUse1,m){m.impl("bbox_loss16",&ptdlprim::bbox_loss16);}
TORCH_LIBRARY_IMPL(pytorch_ocl,AutogradPrivateUse1,m){m.impl("bbox_loss16",&ptdlprim::bbox_loss16);}

TORCH_LIBRARY_FRAGMENT(pytorch_ocl,m) {
    m.def("bbox_loss_l1(Tensor dist, Tensor boxes, Tensor anchors, Tensor target, Tensor scores, Tensor normalizer, Tensor mask, Tensor extent, Tensor stride) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(pytorch_ocl,PrivateUse1,m) {m.impl("bbox_loss_l1",&ptdlprim::bbox_loss_l1);}
TORCH_LIBRARY_IMPL(pytorch_ocl,AutogradPrivateUse1,m) {m.impl("bbox_loss_l1",&ptdlprim::bbox_loss_l1);}
