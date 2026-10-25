#include "batch_pointwise.h"
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <iostream>

namespace ptdlprim {
bool batch_pointwise(at::TensorList a,at::TensorList b,at::TensorList outputs,
                     double w0,double w1,c10::optional<at::Tensor> scalar,
                     BatchExpression expression,bool bump_versions)
{
    if(a.empty() || a.size()!=b.size() || a.size()!=outputs.size())return false;
    auto device=a[0].device();
    if(device.type()!=OpenCLDeviceType)return false;
    auto valid=[&](const at::Tensor &t) {
        return t.layout()==at::kStrided && t.device()==device && t.scalar_type()==at::kFloat &&
               t.is_contiguous() && (!at::GradMode::is_enabled() || !t.requires_grad());
    };
    std::unordered_map<c10::StorageImpl *,size_t> writes;
    for(size_t i=0;i<a.size();i++) {
        if(!valid(a[i]) || !valid(b[i]) || !valid(outputs[i]) ||
           a[i].sizes()!=b[i].sizes() || a[i].sizes()!=outputs[i].sizes())return false;
        if(!writes.emplace(outputs[i].storage().unsafeGetStorageImpl(),i).second)return false;
    }
    auto read_is_safe=[&](const at::Tensor &t,size_t i) {
        auto it=writes.find(t.storage().unsafeGetStorageImpl());
        return it==writes.end() || (it->second==i && t.storage_offset()==outputs[i].storage_offset());
    };
    for(size_t i=0;i<a.size();i++)if(!read_is_safe(a[i],i) || !read_is_safe(b[i],i))return false;
    if(scalar && (!valid(*scalar) || scalar->dim()!=0 || scalar->numel()!=1 ||
                  writes.count(scalar->storage().unsafeGetStorageImpl())))return false;
    auto q=getExecutionContext(a[0]);dlprim::Context ctx(q);
    size_t arg_limit=ctx.device().getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>();
    if(arg_limit<80)return false;
    size_t batch=std::min(size_t(16),(arg_limit-24)/56);
    if(!batch)return false;
    size_t wg=1,limit=std::min(size_t(128),ctx.device().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    limit=std::min(limit,ctx.device().getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0]);
    while(wg*2<=limit)wg*=2;
    static std::mutex mutex;
    static std::map<std::tuple<cl_context,cl_device_id,int,size_t>,cl::Program> programs;
    for(size_t begin=0;begin<a.size();begin+=batch) {
        size_t count=std::min(batch,a.size()-begin);cl::Program program;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto key=std::make_tuple(ctx.context()(),ctx.device()(),int(expression),count);
            auto it=programs.find(key);
            if(it==programs.end()) {
                std::ostringstream code;
                code<<"#pragma OPENCL FP_CONTRACT OFF\n__kernel void batched_pointwise(";
                for(size_t i=0;i<count;i++)code<<"__global const float *a"<<i<<",ulong ao"<<i
                    <<",__global const float *b"<<i<<",ulong bo"<<i
                    <<",__global float *y"<<i<<",ulong yo"<<i<<",ulong n"<<i<<",";
                code<<"float w0,float w1,__global const float *scalar,ulong scalar_offset){"
                    <<"ulong g=get_group_id(0),lane=get_local_id(0),width=get_local_size(0);"
                    <<"__global const float *a=a0,*b=b0;__global float *y=y0;ulong ao=0,bo=0,yo=0,n=0;switch(g){";
                for(size_t i=0;i<count;i++)code<<"case "<<i<<":a=a"<<i<<";b=b"<<i<<";y=y"<<i
                    <<";ao=ao"<<i<<";bo=bo"<<i<<";yo=yo"<<i<<";n=n"<<i<<";break;";
                code<<"}for(ulong j=lane;j<n;j+=width){float x=a[ao+j],v=b[bo+j];";
                switch(expression) {
                case BatchExpression::ema:
                    code<<"volatile float previous=x*w0;volatile float current=v*w1;y[yo+j]=previous+current;";break;
                case BatchExpression::add:
                    code<<"volatile float scaled=v*w0;y[yo+j]=x+scaled;";break;
                case BatchExpression::mul_scalar:code<<"y[yo+j]=x*w0;";break;
                case BatchExpression::mul_tensor:code<<"y[yo+j]=x*scalar[scalar_offset];";break;
                }
                code<<"}}";
                cl::Program built(ctx.context(),code.str());
                try {built.build({ctx.device()});}
                catch(...) {std::cerr<<built.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device());throw;}
                it=programs.emplace(key,built).first;
            }
            program=it->second;
        }
        cl::Kernel kernel(program,"batched_pointwise");int arg=0;
        for(size_t i=0;i<count;i++) {
            auto index=begin+i;
            kernel.setArg(arg++,buffer_from_tensor(a[index]));kernel.setArg(arg++,cl_ulong(a[index].storage_offset()));
            kernel.setArg(arg++,buffer_from_tensor(b[index]));kernel.setArg(arg++,cl_ulong(b[index].storage_offset()));
            kernel.setArg(arg++,buffer_from_tensor(outputs[index]));kernel.setArg(arg++,cl_ulong(outputs[index].storage_offset()));
            kernel.setArg(arg++,cl_ulong(outputs[index].numel()));
        }
        kernel.setArg(arg++,cl_float(w0));kernel.setArg(arg++,cl_float(w1));
        auto coefficient=scalar ? *scalar : a[begin];
        kernel.setArg(arg++,buffer_from_tensor(coefficient));kernel.setArg(arg++,cl_ulong(coefficient.storage_offset()));
        q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(count*wg),cl::NDRange(wg),q.events(),q.event("batched_pointwise"));
    }
    if(bump_versions)for(auto &t:outputs)t.unsafeGetTensorImpl()->bump_version();
    sync_if_needed(device);return true;
}
}

namespace ptdlprim {
void foreach_add_list_(at::TensorList a,at::TensorList b,const at::Scalar &alpha)
{
    GUARD;
    if(!alpha.isComplex() && batch_pointwise(a,b,a,alpha.toDouble(),0,c10::nullopt,BatchExpression::add,true))return;
    at::native::foreach_tensor_add_list_kernel_slow_(a,b,alpha);
}
std::vector<at::Tensor> foreach_add_list(at::TensorList a,at::TensorList b,const at::Scalar &alpha)
{
    GUARD;
    std::vector<at::Tensor> outputs;
    // Restrict speculative allocation to the valid FP32 candidate path.
    bool candidate=!a.empty() && a.size()==b.size() && !alpha.isComplex();
    for(size_t i=0;candidate && i<a.size();i++) {
        candidate=a[i].device().type()==OpenCLDeviceType && a[i].device()==b[i].device() &&
                  a[i].scalar_type()==at::kFloat && b[i].scalar_type()==at::kFloat &&
                  a[i].is_contiguous() && b[i].is_contiguous() && a[i].sizes()==b[i].sizes() &&
                  (!at::GradMode::is_enabled() || (!a[i].requires_grad() && !b[i].requires_grad()));
    }
    if(candidate) {
        for(auto &t:a)outputs.push_back(at::empty_like(t));
        if(batch_pointwise(a,b,outputs,alpha.toDouble(),0,c10::nullopt,BatchExpression::add,false))return outputs;
    }
    return at::native::foreach_tensor_add_list_kernel_slow(a,b,alpha);
}
void foreach_mul_scalar_(at::TensorList a,const at::Scalar &value)
{
    GUARD;
    if(!value.isComplex() && batch_pointwise(a,a,a,value.toDouble(),0,c10::nullopt,BatchExpression::mul_scalar,true))return;
    at::native::foreach_tensor_mul_scalar_kernel_slow_(a,value);
}
void foreach_mul_tensor_(at::TensorList a,const at::Tensor &value)
{
    GUARD;
    if(batch_pointwise(a,a,a,0,0,value,BatchExpression::mul_tensor,true))return;
    at::native::foreach_tensor_mul_tensor_kernel_slow_(a,value);
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
    m.impl("aten::_foreach_add_.List",&ptdlprim::foreach_add_list_);
    m.impl("aten::_foreach_add.List",&ptdlprim::foreach_add_list);
    m.impl("aten::_foreach_mul_.Scalar",&ptdlprim::foreach_mul_scalar_);
    m.impl("aten::_foreach_mul_.Tensor",&ptdlprim::foreach_mul_tensor_);
}
