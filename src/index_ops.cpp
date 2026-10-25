#include "utils.h"
#include <ATen/InferSize.h>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>
#include <iostream>
#include <limits>

namespace ptdlprim {
using namespace at;
static cl::Program index_program(dlprim::Context &ctx,const std::string &code)
{
    static std::mutex mutex;
    static std::map<std::tuple<cl_context,cl_device_id,std::string>,cl::Program> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto key=std::make_tuple(ctx.context()(),ctx.device()(),code);
    auto it=cache.find(key);
    if(it==cache.end()) {
        cl::Program program(ctx.context(),code);
        try {program.build({ctx.device()});}
        catch(...) {std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device());throw;}
        it=cache.emplace(key,program).first;
    }
    return it->second;
}
static bool supported_index_type(const Tensor &t)
{
    auto d=t.scalar_type();
    return d==kFloat || d==kLong || d==kInt || d==kBool || d==kByte || d==kChar;
}
Tensor nonzero(const Tensor &self)
{
    GUARD;
    if(!supported_index_type(self) || self.numel()>std::numeric_limits<int>::max())
        return at::nonzero(self.cpu()).to(self.device());
    if(self.numel()==0)return at::empty({0,self.dim()},self.options().dtype(kLong));
    auto x=self.contiguous();
    auto q=getExecutionContext(self);dlprim::Context ctx(q);
    size_t width=1,limit=std::min(size_t(128),ctx.device().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    limit=std::min(limit,ctx.device().getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0]);
    limit=std::min(limit,size_t(ctx.device().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/sizeof(cl_int)));
    while(width*2<=limit)width*=2;
    int64_t groups=(self.numel()+width-1)/width;
    auto prefix=at::empty({self.numel()},self.options().dtype(kInt));
    auto offsets=at::empty({groups},self.options().dtype(kInt));
    auto total=at::empty({1},self.options().dtype(kLong));
    std::ostringstream source;
    source<<"__kernel void flags(__global const "<<dlprim::data_type_to_opencl_type(todp(self.dtype()))
          <<" *x,ulong xo,ulong n,__global int *p,__global int *o,__local int *s){"
          <<"ulong g=get_global_id(0),l=get_local_id(0),w=get_local_size(0);"
          <<"int flag=g<n && x[xo+g]!=0;s[l]=flag;barrier(CLK_LOCAL_MEM_FENCE);"
          <<"for(ulong d=1;d<w;d*=2){int add=l>=d?s[l-d]:0;barrier(CLK_LOCAL_MEM_FENCE);s[l]+=add;barrier(CLK_LOCAL_MEM_FENCE);}"
          <<"if(g<n)p[g]=s[l]-flag;if(l==w-1)o[get_group_id(0)]=s[l];}"
          <<"__kernel void block_offsets(__global int *o,ulong groups,__global long *total){"
          <<"if(get_global_id(0))return;int sum=0;for(ulong j=0;j<groups;j++){int v=o[j];o[j]=sum;sum+=v;}total[0]=(long)sum;}"
          <<"__kernel void coordinates(__global const "<<dlprim::data_type_to_opencl_type(todp(self.dtype()))
          <<" *x,ulong xo,ulong n,__global const int *p,__global const int *o,ulong width,__global long *y";
    for(int d=0;d<self.dim();d++)source<<",ulong size"<<d;
    source<<"){ulong g=get_global_id(0);if(g>=n || x[xo+g]==0)return;ulong row=o[g/width]+p[g],rem=g;";
    for(int d=self.dim()-1;d>=0;d--)source<<"y[row*"<<self.dim()<<"+"<<d<<"]=rem%size"<<d<<";rem/=size"<<d<<";";
    source<<"}";
    auto program=index_program(ctx,source.str());
    cl::Kernel flags(program,"flags");int arg=0;
    flags.setArg(arg++,buffer_from_tensor(x));flags.setArg(arg++,cl_ulong(x.storage_offset()));
    flags.setArg(arg++,cl_ulong(x.numel()));flags.setArg(arg++,buffer_from_tensor(prefix));
    flags.setArg(arg++,buffer_from_tensor(offsets));flags.setArg(arg++,cl::Local(width*sizeof(cl_int)));
    q.queue().enqueueNDRangeKernel(flags,cl::NullRange,cl::NDRange(groups*width),cl::NDRange(width),q.events(),q.event("nonzero_flags"));
    cl::Kernel scan(program,"block_offsets");
    scan.setArg(0,buffer_from_tensor(offsets));scan.setArg(1,cl_ulong(groups));scan.setArg(2,buffer_from_tensor(total));
    q.queue().enqueueNDRangeKernel(scan,cl::NullRange,cl::NDRange(1),cl::NDRange(1),q.events(),q.event("nonzero_block_offsets"));
    // Dynamic output size requires one scalar read, not downloading the mask.
    int64_t count=total.item<int64_t>();
    auto output=at::empty({count,self.dim()},self.options().dtype(kLong));
    if(count && self.dim()) {
        cl::Kernel gather(program,"coordinates");arg=0;
        gather.setArg(arg++,buffer_from_tensor(x));gather.setArg(arg++,cl_ulong(x.storage_offset()));
        gather.setArg(arg++,cl_ulong(x.numel()));gather.setArg(arg++,buffer_from_tensor(prefix));
        gather.setArg(arg++,buffer_from_tensor(offsets));gather.setArg(arg++,cl_ulong(width));
        gather.setArg(arg++,buffer_from_tensor(output));
        for(auto size:self.sizes())gather.setArg(arg++,cl_ulong(size));
        q.queue().enqueueNDRangeKernel(gather,cl::NullRange,cl::NDRange(self.numel()),cl::NullRange,q.events(),q.event("nonzero_coordinates"));
    }
    sync_if_needed(self.device());return output;
}
Tensor & nonzero_out(const Tensor &self,Tensor &out)
{
    TORCH_CHECK(out.scalar_type()==kLong && out.device()==self.device(),"nonzero out must be int64 on the input device");
    auto result=ptdlprim::nonzero(self);out.resize_(result.sizes());out.copy_(result);return out;
}
static Tensor index_cpu(const Tensor &self,const c10::List<c10::optional<Tensor>> &indices)
{
    c10::List<c10::optional<Tensor>> cpu;
    for(size_t i=0;i<indices.size();i++) {auto t=indices.get(i);cpu.push_back(t && t->defined() ? c10::optional<Tensor>(t->cpu()) : c10::nullopt);}
    return at::index(self.cpu(),cpu).to(self.device());
}
Tensor advanced_index(const Tensor &self,const c10::List<c10::optional<Tensor>> &indices)
{
    GUARD;
    if(self.device().type()!=OpenCLDeviceType || !supported_index_type(self) || self.dim()>8)
        return index_cpu(self,indices);
    std::vector<c10::optional<Tensor>> by_axis;
    std::vector<bool> bounded;
    int axis=0;
    for(size_t i=0;i<indices.size();i++) {
        auto t=indices.get(i);
        if(!t || !t->defined()) {by_axis.push_back(c10::nullopt);bounded.push_back(false);axis++;continue;}
        if(t->scalar_type()==kBool) {
            if(t->dim()==0)return index_cpu(self,indices);
            TORCH_CHECK(axis+t->dim()<=self.dim(),"too many indices for tensor");
            for(int d=0;d<t->dim();d++)TORCH_CHECK(t->size(d)==self.size(axis+d),"boolean index shape does not match tensor");
            auto coords=ptdlprim::nonzero(t->to(self.device()));
            for(int d=0;d<t->dim();d++){by_axis.push_back(coords.select(1,d).contiguous());bounded.push_back(true);}
            axis+=t->dim();
        }
        else if(t->scalar_type()==kLong || t->scalar_type()==kInt) {
            by_axis.push_back(t->to(self.device(),kLong).contiguous());bounded.push_back(false);axis++;
        }
        else return index_cpu(self,indices);
    }
    TORCH_CHECK(axis<=self.dim(),"too many indices for tensor");
    while(int(by_axis.size())<self.dim()){by_axis.push_back(c10::nullopt);bounded.push_back(false);}
    std::vector<int> indexed,unindexed;
    std::vector<int64_t> broadcast;
    for(int d=0;d<self.dim();d++) {
        if(by_axis[d]) {indexed.push_back(d);broadcast=at::infer_size(broadcast,by_axis[d]->sizes());}
        else unindexed.push_back(d);
    }
    if(indexed.empty())return self.clone();
    bool adjacent=true;
    for(size_t i=1;i<indexed.size();i++)adjacent &= indexed[i]==indexed[i-1]+1;
    int prefix=adjacent ? indexed[0] : 0;
    std::vector<int64_t> shape;
    if(adjacent)for(int d=0;d<prefix;d++)shape.push_back(self.size(d));
    shape.insert(shape.end(),broadcast.begin(),broadcast.end());
    for(auto d:unindexed)if(!adjacent || d>=prefix)shape.push_back(self.size(d));
    if(shape.size()>8 || broadcast.size()>8)return index_cpu(self,indices);
    auto out=at::empty(shape,self.options());if(!out.numel())return out;
    bool check_bounds=false;for(auto d:indexed)check_bounds |= !bounded[d];
    auto error=at::zeros({1},self.options().dtype(kInt));
    auto q=getExecutionContext(self);dlprim::Context ctx(q);
    size_t bytes=48+16*self.dim()+8*shape.size()+indexed.size()*(16+8*broadcast.size());
    if(bytes>ctx.device().getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>())return index_cpu(self,indices);
    std::ostringstream code;
    code<<"__kernel void index_gather(__global const "<<dlprim::data_type_to_opencl_type(todp(self.dtype()))
        <<" *x,ulong xo,__global "<<dlprim::data_type_to_opencl_type(todp(self.dtype()))
        <<" *y,ulong yo,__global int *error,ulong n";
    for(int d=0;d<self.dim();d++)code<<",ulong s"<<d<<",ulong stride"<<d;
    for(size_t d=0;d<shape.size();d++)code<<",ulong os"<<d;
    for(size_t j=0;j<indexed.size();j++) {
        code<<",__global const long *i"<<j<<",ulong io"<<j;
        for(size_t d=0;d<broadcast.size();d++)code<<",ulong is"<<j<<"_"<<d;
    }
    code<<"){ulong g=get_global_id(0);if(g>=n)return;ulong rem=g,address=xo;int valid=1;";
    for(int d=int(shape.size())-1;d>=0;d--)code<<"ulong c"<<d<<"=rem%os"<<d<<";rem/=os"<<d<<";";
    for(size_t j=0;j<unindexed.size();j++) {
        int d=unindexed[j],position=(adjacent && d<prefix) ? int(j) : int(j+broadcast.size());
        code<<"address+=c"<<position<<"*stride"<<d<<";";
    }
    for(size_t j=0;j<indexed.size();j++) {
        int d=indexed[j];code<<"ulong p"<<j<<"=io"<<j<<";";
        for(size_t b=0;b<broadcast.size();b++)code<<"p"<<j<<"+=c"<<(prefix+b)<<"*is"<<j<<"_"<<b<<";";
        code<<"long v"<<j<<"=i"<<j<<"[p"<<j<<"];if(v"<<j<<"<0)v"<<j<<"+=s"<<d<<";";
        if(!bounded[d])code<<"valid &= v"<<j<<">=0 && v"<<j<<"<(long)s"<<d<<";";
        code<<"address+=(ulong)v"<<j<<"*stride"<<d<<";";
    }
    code<<"if(!valid){atomic_or((volatile __global int *)error,1);return;}y[yo+g]=x[address];}";
    auto program=index_program(ctx,code.str());cl::Kernel kernel(program,"index_gather");int arg=0;
    kernel.setArg(arg++,buffer_from_tensor(self));kernel.setArg(arg++,cl_ulong(self.storage_offset()));
    kernel.setArg(arg++,buffer_from_tensor(out));kernel.setArg(arg++,cl_ulong(out.storage_offset()));
    kernel.setArg(arg++,buffer_from_tensor(error));kernel.setArg(arg++,cl_ulong(out.numel()));
    for(int d=0;d<self.dim();d++){kernel.setArg(arg++,cl_ulong(self.size(d)));kernel.setArg(arg++,cl_ulong(self.stride(d)));}
    for(auto size:shape)kernel.setArg(arg++,cl_ulong(size));
    for(auto d:indexed) {
        auto &t=*by_axis[d];kernel.setArg(arg++,buffer_from_tensor(t));kernel.setArg(arg++,cl_ulong(t.storage_offset()));
        for(size_t b=0;b<broadcast.size();b++) {
            int td=int(b)+t.dim()-int(broadcast.size());
            kernel.setArg(arg++,cl_ulong(td<0 || t.size(td)==1 ? 0 : t.stride(td)));
        }
    }
    q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(out.numel()),cl::NullRange,q.events(),q.event("index_gather"));
    if(check_bounds)TORCH_CHECK(error.item<int>()==0,"index out of bounds");
    sync_if_needed(self.device());return out;
}
Tensor & advanced_index_out(const Tensor &self,const c10::List<c10::optional<Tensor>> &indices,Tensor &out)
{
    TORCH_CHECK(out.device()==self.device() && out.scalar_type()==self.scalar_type(),"index output device/dtype mismatch");
    auto value=advanced_index(self,indices);out.resize_(value.sizes());out.copy_(value);return out;
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
    m.impl("aten::nonzero",&ptdlprim::nonzero);
    m.impl("aten::nonzero.out",&ptdlprim::nonzero_out);
    m.impl("aten::index.Tensor",&ptdlprim::advanced_index);
    m.impl("aten::index.Tensor_out",&ptdlprim::advanced_index_out);
}
