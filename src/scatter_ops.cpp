#include "utils.h"
#include <dlprim/core/pointwise.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <cstdlib>
#include <limits>

namespace ptdlprim {
using namespace at;

// One thread owns each destination element: repeated indices need no atomics.
// Suited to target assignment's tiny scatter dimension, not a universal fast
// scatter implementation. Other shapes retain PyTorch's CPU fallback below.
Tensor & scatter_target_out(const Tensor &self,int64_t dim,const Tensor &index,
                           const c10::optional<Tensor> &src,const Scalar &value,
                           Tensor &out)
{
    GUARD;
    dim=at::maybe_wrap_dim(dim,self.dim());
    bool add=src.has_value();
    // Preserve an existing output view when its logical shape already matches.
    auto resize_output=[&](){if(out.sizes()!=self.sizes())out.resize_as_(self);};
    TORCH_CHECK(index.scalar_type()==kLong,"scatter index must be int64");
    TORCH_CHECK(index.dim()==self.dim(),"scatter index rank mismatch");
    if(add) TORCH_CHECK(src->dim()==self.dim() && src->scalar_type()==self.scalar_type(),"scatter source mismatch");
    for(int d=0;d<self.dim();d++) {
        if(d!=dim) TORCH_CHECK(index.size(d)<=self.size(d),"scatter index shape mismatch");
        if(add) TORCH_CHECK(index.size(d)<=src->size(d),"scatter source shape mismatch");
    }
    if(index.numel()==0) {resize_output();out.copy_(self);return out;}
    // Avoid an O(destination * index-axis) algorithm on large scatters.
    if(self.dim()==0 || self.numel()==0 || index.size(dim)>32 || (self.scalar_type()!=kFloat && self.scalar_type()!=kChar && self.scalar_type()!=kLong)) {
        auto cpu=add ? self.cpu().scatter_add(dim,index.cpu(),src->cpu()) : self.cpu().scatter(dim,index.cpu(),value);
        resize_output();out.copy_(cpu);return out;
    }
    // r600 subword shader stores can corrupt an entire word. Compute small
    // int8 scatters in FP32 (all intermediate integers are exact here), then
    // use the packed byte conversion path to preserve native tensor storage.
    const char *packed=std::getenv("DLPRIM_PACKED_BYTE_OUTPUTS");
    if(self.scalar_type()==kChar && packed && std::string(packed)=="1") {
        if(add) TORCH_CHECK(!src->is_alias_of(out),"scatter source must not alias output");
        auto wide=self.to(kFloat);
        auto result=at::empty_like(wide);
        c10::optional<Tensor> wide_src=c10::nullopt;
        if(add) wide_src=src->to(kFloat);
        scatter_target_out(wide,dim,index,wide_src,value,result);
        resize_output();
        out.copy_(result.to(kChar));
        return out;
    }
    auto x=self.contiguous(), idx=index.contiguous();
    Tensor source=add ? src->contiguous() : x;
    // Disallow source aliasing just as native scatter does.
    if(add) TORCH_CHECK(!src->is_alias_of(out),"scatter source must not alias output");
    resize_output();
    auto output=out.contiguous();
    auto q=getExecutionContext(self);dlprim::Context ctx(q);
    bool legacy32=ctx.device().getInfo<CL_DEVICE_ADDRESS_BITS>()==32;
    if(legacy32) {
        auto fits=[](int64_t v){return v>=0 && uint64_t(v)<=std::numeric_limits<cl_uint>::max();};
        bool bounded=true;
        for(auto const &t:{x,idx,source,output}) {
            bounded &= fits(t.numel()) && fits(t.storage_offset()) &&
                       uint64_t(t.storage_offset())+uint64_t(t.numel())<=std::numeric_limits<cl_uint>::max();
            for(auto v:t.sizes())bounded &= fits(v);
            for(auto v:t.strides())bounded &= fits(v);
        }
        if(!bounded) {
            auto cpu=add ? self.cpu().scatter_add(dim,index.cpu(),src->cpu()) : self.cpu().scatter(dim,index.cpu(),value);
            out.copy_(cpu);return out;
        }
    }
    auto error=at::zeros({legacy32 ? self.numel() : int64_t(1)},self.options().dtype(kInt));
    std::string type=dlprim::data_type_to_opencl_type(todp(self.dtype()));
    std::ostringstream code;
    if(legacy32)code<<"inline __attribute__((always_inline)) uint2 scatter_divmod(uint n,uint d){uint q=0,r=0;for(int b=31;b>=0;--b){uint carry=r>>31;r=(r<<1)|((n>>b)&1u);uint take=carry | (r>=d);r-=d*take;q|=take<<b;}return (uint2)(q,r);}";
    code << "__kernel void target_scatter(__global const " <<type<<" *x, ulong xo, __global const "<<(legacy32 ? "uint2" : "long")<<" *idx, ulong io, __global const "<<type<<" *src, ulong so, __global "<<type<<" *y, ulong yo, __global int *err, ulong n, ulong axis_size, ulong count, "<<type<<" value";
    for(int d=0;d<self.dim();d++) code<<", ulong size"<<d<<", ulong isize"<<d<<", ulong istride"<<d<<", ulong sstride"<<d;
    code << ") { ulong pos=get_global_id(0); if(pos>=n)return; ulong rem=pos, ib=0,sb=0; int inside=1;";
    if(legacy32)code<<"err[pos]=0;";
    for(int d=self.dim()-1;d>=0;d--) {
        if(legacy32)code<<"uint2 qr"<<d<<"=scatter_divmod(rem,size"<<d<<");uint c"<<d<<"=qr"<<d<<".y;rem=qr"<<d<<".x;";
        else code<<"ulong c"<<d<<"=rem%size"<<d<<"; rem/=size"<<d<<";";
        if(d!=dim) code<<"inside &= c"<<d<<"<isize"<<d<<"; ib+=c"<<d<<"*istride"<<d<<"; sb+=c"<<d<<"*sstride"<<d<<";";
    }
    code<<type<<" v=x[xo+pos]; if(inside) { for(ulong k=0;k<count;k++){ ";
    if(legacy32) {
        bool little=ctx.device().getInfo<CL_DEVICE_ENDIAN_LITTLE>();
        code<<"uint2 words=idx[io+ib+k*istride"<<dim<<"];uint j=words."<<(little ? "x" : "y")<<";if(words."<<(little ? "y" : "x")<<"!=0u || j>=axis_size){err[pos]=1;continue;}";
    }
    else code<<"long j=idx[io+ib+k*istride"<<dim<<"];if(j<0 || j>=axis_size){atomic_or((volatile __global int *)err,1);continue;}";
    code<<"if(j==c"<<dim<<") ";
    code<<(add ? "v+=src[so+sb+k*sstride"+std::to_string(dim)+"];" : "v=value;");
    code<<"}} y[yo+pos]=v;}";
    std::string kernel_source=code.str();
    if(legacy32)for(size_t pos=0;(pos=kernel_source.find("ulong",pos))!=std::string::npos;pos+=4)kernel_source.replace(pos,5,"uint");
    static std::mutex mutex;
    static std::map<std::pair<cl_context,std::string>,cl::Program> programs;
    cl::Program program;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto key=std::make_pair(ctx.context()(),kernel_source);
        auto it=programs.find(key);
        if(it==programs.end()) {
            cl::Program built(ctx.context(),kernel_source);
            try {built.build({ctx.device()});}
            catch(...) {std::cerr<<built.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device());throw;}
            it=programs.emplace(key,built).first;
        }
        program=it->second;
    }
    cl::Kernel kernel(program,"target_scatter");int p=0;
    auto set_index=[&](cl_ulong v){if(legacy32)kernel.setArg(p++,cl_uint(v));else kernel.setArg(p++,v);};
    kernel.setArg(p++,buffer_from_tensor(x));set_index(x.storage_offset());
    kernel.setArg(p++,buffer_from_tensor(idx));set_index(idx.storage_offset());
    kernel.setArg(p++,buffer_from_tensor(source));set_index(source.storage_offset());
    kernel.setArg(p++,buffer_from_tensor(output));set_index(output.storage_offset());
    kernel.setArg(p++,buffer_from_tensor(error));set_index(self.numel());
    set_index(self.size(dim));set_index(index.size(dim));
    dlprim::core::bind_as_dtype(kernel,p,value.toDouble(),todp(self.dtype()));
    for(int d=0;d<self.dim();d++) {
        set_index(self.size(d));set_index(idx.size(d));
        set_index(idx.stride(d));set_index(source.stride(d));
    }
    q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(self.numel()),cl::NullRange,q.events(),q.event("target_scatter"));
    TORCH_CHECK((legacy32 ? error.max().item<int>() : error.item<int>())==0,"scatter index out of bounds");
    if(!out.is_contiguous())out.copy_(output);
    sync_if_needed(self.device());return out;
}
Tensor & scatter_add_out(const Tensor &self,int64_t dim,const Tensor &index,const Tensor &src,Tensor &out)
{return scatter_target_out(self,dim,index,src,0,out);}
Tensor & scatter_value_out(const Tensor &self,int64_t dim,const Tensor &index,const Scalar &value,Tensor &out)
{return scatter_target_out(self,dim,index,c10::nullopt,value,out);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
 m.impl("aten::scatter_add.out",&ptdlprim::scatter_add_out);
 m.impl("aten::scatter.value_out",&ptdlprim::scatter_value_out);
}
