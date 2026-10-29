#include "utils.h"
#include <dlprim/core/pointwise.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <cstdlib>

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
    TORCH_CHECK(index.scalar_type()==kLong,"scatter index must be int64");
    TORCH_CHECK(index.dim()==self.dim(),"scatter index rank mismatch");
    if(add) TORCH_CHECK(src->dim()==self.dim() && src->scalar_type()==self.scalar_type(),"scatter source mismatch");
    for(int d=0;d<self.dim();d++) {
        if(d!=dim) TORCH_CHECK(index.size(d)<=self.size(d),"scatter index shape mismatch");
        if(add) TORCH_CHECK(index.size(d)<=src->size(d),"scatter source shape mismatch");
    }
    if(index.numel()==0) {out.resize_as_(self);out.copy_(self);return out;}
    // Avoid an O(destination * index-axis) algorithm on large scatters.
    if(self.dim()==0 || index.size(dim)>32 || (self.scalar_type()!=kFloat && self.scalar_type()!=kChar && self.scalar_type()!=kLong)) {
        auto cpu=add ? self.cpu().scatter_add(dim,index.cpu(),src->cpu()) : self.cpu().scatter(dim,index.cpu(),value);
        out.resize_as_(self);out.copy_(cpu);return out;
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
        out.resize_as_(self);
        out.copy_(result.to(kChar));
        return out;
    }
    auto x=self.contiguous(), idx=index.contiguous();
    Tensor source=add ? src->contiguous() : x;
    // Disallow source aliasing just as native scatter does.
    if(add) TORCH_CHECK(!src->is_alias_of(out),"scatter source must not alias output");
    out.resize_as_(self);
    auto output=out.contiguous();
    auto error=at::zeros({1},self.options().dtype(kInt));
    std::string type=dlprim::data_type_to_opencl_type(todp(self.dtype()));
    std::ostringstream code;
    code << "__kernel void target_scatter(__global const " <<type<<" *x, ulong xo, __global const long *idx, ulong io, __global const "<<type<<" *src, ulong so, __global "<<type<<" *y, ulong yo, __global int *err, ulong n, ulong axis_size, ulong count, "<<type<<" value";
    for(int d=0;d<self.dim();d++) code<<", ulong size"<<d<<", ulong isize"<<d<<", ulong istride"<<d<<", ulong sstride"<<d;
    code << ") { ulong pos=get_global_id(0); if(pos>=n)return; ulong rem=pos, ib=0,sb=0; int inside=1;";
    for(int d=self.dim()-1;d>=0;d--) {
        code<<"ulong c"<<d<<"=rem%size"<<d<<"; rem/=size"<<d<<";";
        if(d!=dim) code<<"inside &= c"<<d<<"<isize"<<d<<"; ib+=c"<<d<<"*istride"<<d<<"; sb+=c"<<d<<"*sstride"<<d<<";";
    }
    code<<type<<" v=x[xo+pos]; if(inside) { for(ulong k=0;k<count;k++){ long j=idx[io+ib+k*istride"<<dim<<"]; if(j<0 || j>=axis_size) {atomic_or((volatile __global int *)err,1);continue;} if(j==c"<<dim<<") ";
    code<<(add ? "v+=src[so+sb+k*sstride"+std::to_string(dim)+"];" : "v=value;");
    code<<"}} y[yo+pos]=v;}";
    auto q=getExecutionContext(self);dlprim::Context ctx(q);
    static std::mutex mutex;
    static std::map<std::pair<cl_context,std::string>,cl::Program> programs;
    cl::Program program;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto key=std::make_pair(ctx.context()(),code.str());
        auto it=programs.find(key);
        if(it==programs.end()) {
            cl::Program built(ctx.context(),code.str());
            try {built.build({ctx.device()});}
            catch(...) {std::cerr<<built.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device());throw;}
            it=programs.emplace(key,built).first;
        }
        program=it->second;
    }
    cl::Kernel kernel(program,"target_scatter");int p=0;
    kernel.setArg(p++,buffer_from_tensor(x));kernel.setArg(p++,cl_ulong(x.storage_offset()));
    kernel.setArg(p++,buffer_from_tensor(idx));kernel.setArg(p++,cl_ulong(idx.storage_offset()));
    kernel.setArg(p++,buffer_from_tensor(source));kernel.setArg(p++,cl_ulong(source.storage_offset()));
    kernel.setArg(p++,buffer_from_tensor(output));kernel.setArg(p++,cl_ulong(output.storage_offset()));
    kernel.setArg(p++,buffer_from_tensor(error));kernel.setArg(p++,cl_ulong(self.numel()));
    kernel.setArg(p++,cl_ulong(self.size(dim)));kernel.setArg(p++,cl_ulong(index.size(dim)));
    dlprim::core::bind_as_dtype(kernel,p,value.toDouble(),todp(self.dtype()));
    for(int d=0;d<self.dim();d++) {
        kernel.setArg(p++,cl_ulong(self.size(d)));kernel.setArg(p++,cl_ulong(idx.size(d)));
        kernel.setArg(p++,cl_ulong(idx.stride(d)));kernel.setArg(p++,cl_ulong(source.stride(d)));
    }
    q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(self.numel()),cl::NullRange,q.events(),q.event("target_scatter"));
    TORCH_CHECK(error.item<int>()==0,"scatter index out of bounds");
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
