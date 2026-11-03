#include "utils.h"
#include <ATen/ops/stack_native.h>
#include <map>
#include <mutex>
#include <sstream>
#include <iostream>

namespace ptdlprim {
std::vector<at::Tensor> foreach_norm(const at::TensorList tensors,const at::Scalar &ord,c10::optional<at::ScalarType> dtype)
{
    GUARD;
    TORCH_CHECK(!tensors.empty(),"Tensor list must have at least one tensor");
    auto dev=tensors[0].device();
    bool eligible=ord.toDouble()==2.0 && (!dtype || *dtype==at::kFloat);
    for(auto &t:tensors) {
        eligible &= t.device()==dev && t.scalar_type()==at::kFloat && t.is_contiguous();
        // Keep independent scalar outputs for differentiable foreach calls;
        // clipping is no_grad and can safely use packed scalar views.
        eligible &= !at::GradMode::is_enabled() || !t.requires_grad();
    }
    if(!eligible) {
        std::vector<at::Tensor> result;
        for(auto &t:tensors)result.push_back(at::linalg_vector_norm(t,ord,c10::nullopt,false,dtype));
        return result;
    }
    auto packed=at::empty({int64_t(tensors.size())},tensors[0].options());
    auto q=getExecutionContext(tensors[0]);dlprim::Context ctx(q);
    size_t limit=std::min(size_t(128),ctx.device().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    limit=std::min(limit,ctx.device().getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0]);
    limit=std::min(limit,size_t(ctx.device().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/sizeof(float)));
    size_t wg=1;while(wg*2<=limit)wg*=2;
    // 16 buffer/offset/size triples fit even a 512-byte kernel-argument budget.
    constexpr size_t batch=16;
    static std::mutex mutex;
    static std::map<std::pair<cl_context,size_t>,cl::Program> programs;
    for(size_t begin=0;begin<tensors.size();begin+=batch) {
        size_t count=std::min(batch,tensors.size()-begin);
        cl::Program program;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto key=std::make_pair(ctx.context()(),count);
            auto it=programs.find(key);
            if(it==programs.end()) {
                std::ostringstream code;
                code<<"#pragma OPENCL FP_CONTRACT OFF\n__kernel void batched_norm(";
                for(size_t i=0;i<count;i++)code<<"__global const float *a"<<i<<", ulong o"<<i<<", ulong n"<<i<<",";
                code<<"__global float *out, ulong out_offset, __local float *scratch) {"
                    <<"ulong g=get_group_id(0), lane=get_local_id(0), width=get_local_size(0);"
                    <<"__global const float *data=a0; ulong offset=0,n=0; switch(g){";
                for(size_t i=0;i<count;i++)code<<"case "<<i<<":data=a"<<i<<";offset=o"<<i<<";n=n"<<i<<";break;";
                code<<"} float sum=0,correction=0;int has_inf=0,has_nan=0;for(ulong j=lane;j<n;j+=width){float v=data[offset+j],term=v*v;if(isnan(term)){has_nan=1;}else if(isinf(term)){has_inf=1;}else if(!has_inf){float adjusted=term-correction;float next=sum+adjusted;if(isinf(next)){has_inf=1;}else{correction=(next-sum)-adjusted;sum=next;}}}sum=has_nan?NAN:(has_inf?INFINITY:sum);"
                    <<"scratch[lane]=sum;barrier(CLK_LOCAL_MEM_FENCE);"
                    <<"for(ulong stride=width/2;stride;stride/=2){if(lane<stride)scratch[lane]+=scratch[lane+stride];barrier(CLK_LOCAL_MEM_FENCE);}"
                    <<"if(lane==0)out[out_offset+g]=sqrt(scratch[0]);}";
                cl::Program built(ctx.context(),code.str());
                try {built.build({ctx.device()});}
                catch(...) {std::cerr<<built.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device());throw;}
                it=programs.emplace(key,built).first;
            }
            program=it->second;
        }
        cl::Kernel kernel(program,"batched_norm");int arg=0;
        for(size_t i=0;i<count;i++) {
            auto &t=tensors[begin+i];
            kernel.setArg(arg++,buffer_from_tensor(t));
            kernel.setArg(arg++,cl_ulong(t.storage_offset()));kernel.setArg(arg++,cl_ulong(t.numel()));
        }
        kernel.setArg(arg++,buffer_from_tensor(packed));
        kernel.setArg(arg++,cl_ulong(packed.storage_offset()+begin));
        kernel.setArg(arg++,cl::Local(wg*sizeof(float)));
        q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(count*wg),cl::NDRange(wg),q.events(),q.event("batched_norm"));
    }
    sync_if_needed(dev);
    std::vector<at::Tensor> result;
    for(size_t i=0;i<tensors.size();i++)result.push_back(packed.select(0,i));
    return result;
}

// Native clipping stacks the scalar norms. Recognize adjacent scalar views
// and copy their packed storage once rather than launch a copy for each scalar.
at::Tensor stack_norms(at::TensorList tensors,int64_t dim)
{
    GUARD;
    bool packed=!tensors.empty() && (dim==0 || dim==-1);
    if(packed) {
        auto &first=tensors[0];packed=first.dim()==0;
        for(size_t i=0;packed && i<tensors.size();i++) {
            auto &t=tensors[i];
            packed=t.dim()==0 && t.device()==first.device() && t.scalar_type()==first.scalar_type() &&
                   t.is_alias_of(first) && t.storage_offset()==first.storage_offset()+int64_t(i);
        }
        if(packed)return first.as_strided({int64_t(tensors.size())},{1},first.storage_offset()).clone();
    }
    return at::native::stack(tensors,dim);
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
    m.impl("aten::_foreach_norm.Scalar",&ptdlprim::foreach_norm);
    m.impl("aten::stack",&ptdlprim::stack_norms);
}
