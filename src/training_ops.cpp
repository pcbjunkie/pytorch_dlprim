#include "utils.h"
#include <ATen/MemoryOverlap.h>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>
#include <iostream>
namespace ptdlprim {
using namespace at;
static cl::Program program_for(dlprim::Context &ctx,const std::string &code) {
 static std::mutex mutex;static std::map<std::tuple<cl_context,cl_device_id,std::string>,cl::Program> cache;
 std::lock_guard<std::mutex> lock(mutex);auto key=std::make_tuple(ctx.context()(),ctx.device()(),code);
 auto it=cache.find(key);if(it==cache.end()){cl::Program p(ctx.context(),code);try{p.build({ctx.device()});}catch(...){std::cerr<<p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device());throw;}it=cache.emplace(key,p).first;}return it->second;
}
Tensor &addmv_gpu(const Tensor &self,const Tensor &mat,const Tensor &vec,const Scalar &beta,const Scalar &alpha,Tensor &out) {
 GUARD;
 TORCH_CHECK(mat.dim()==2 && vec.dim()==1 && mat.size(1)==vec.size(0),"addmv matrix/vector shape mismatch");
 TORCH_CHECK(self.dim()<=1 && (self.numel()==1 || self.size(0)==mat.size(0)),"addmv input is not broadcastable");
 TORCH_CHECK(self.device()==mat.device() && vec.device()==mat.device() && out.device()==mat.device(),"addmv device mismatch");
 TORCH_CHECK(self.scalar_type()==mat.scalar_type() && vec.scalar_type()==mat.scalar_type() && out.scalar_type()==mat.scalar_type(),"addmv dtype mismatch");
 auto product=(mat.size(0)==0 || mat.size(1)==0) ? at::zeros({mat.size(0)},mat.options()) : at::mm(mat,vec.reshape({vec.numel(),1})).reshape({mat.size(0)});
 // beta=0 must not propagate NaNs from the ignored input. Materialize before
 // resizing out so out=self and overlapping inputs remain safe.
 auto result=product.mul(alpha);if(!beta.equal(0))result.add_(self,beta);
 out.resize_({mat.size(0)});out.copy_(result);return out;
}
std::tuple<Tensor&,Tensor&> topk_gpu(const Tensor &self,int64_t k,int64_t dim,bool largest,bool sorted,Tensor &values,Tensor &indices) {
 GUARD;dim=at::maybe_wrap_dim(dim,self.dim());
 TORCH_CHECK(k>=0 && k<=(self.dim()?self.size(dim):1),"selected index k out of range");
 TORCH_CHECK(values.device()==self.device() && values.scalar_type()==self.scalar_type() && indices.device()==self.device() && indices.scalar_type()==kLong,"topk output mismatch");
 if(self.scalar_type()!=kFloat && self.scalar_type()!=kLong && self.scalar_type()!=kInt){TORCH_WARN_ONCE("OpenCL topk: unsupported dtype uses CPU");auto r=at::topk(self.cpu(),k,dim,largest,sorted);values.resize_(std::get<0>(r).sizes()).copy_(std::get<0>(r));indices.resize_(std::get<1>(r).sizes()).copy_(std::get<1>(r));return {values,indices};}
 if(self.dim()==0) {values.resize_({}).copy_(self);indices.resize_({}).zero_();return {values,indices};}
 auto shape=self.sizes().vec();if(self.dim())shape[dim]=k;
 auto x=self.contiguous(),v=at::empty(shape,self.options()),idx=at::empty(shape,self.options().dtype(kLong));
 int64_t inner=1,axis=self.dim()?self.size(dim):1;for(int d=dim+1;d<self.dim();d++)inner*=self.size(d);
 int64_t rows=axis?self.numel()/axis:0;
 if(k && rows){auto q=getExecutionContext(self);dlprim::Context ctx(q);auto type=dlprim::data_type_to_opencl_type(todp(self.dtype()));std::ostringstream s;
 s<<"__kernel void select_k(__global const "<<type<<"*x,ulong xo,__global "<<type<<"*v,__global long*i,ulong rows,ulong axis,ulong inner,ulong k,int largest){ulong r=get_global_id(0);if(r>=rows)return;ulong base=(r/inner)*axis*inner+r%inner,ob=(r/inner)*k*inner+r%inner;for(ulong t=0;t<k;t++){long best=-1;"<<type<<" bv=0;for(ulong j=0;j<axis;j++){int used=0;for(ulong p=0;p<t;p++)used|=i[ob+p*inner]==(long)j;if(used)continue;"<<type<<" a=x[xo+base+j*inner];int better=best<0 || (largest?a>bv:a<bv);";
 if(self.scalar_type()==kFloat)s<<"better=best<0 || (isnan(a)?(largest&&!isnan(bv)):(isnan(bv)?!largest:(largest?a>bv:a<bv)));";
 s<<"if(better){best=j;bv=a;}}v[ob+t*inner]=bv;i[ob+t*inner]=best;}}";
 cl::Kernel kernel(program_for(ctx,s.str()),"select_k");int a=0;kernel.setArg(a++,buffer_from_tensor(x));kernel.setArg(a++,cl_ulong(x.storage_offset()));kernel.setArg(a++,buffer_from_tensor(v));kernel.setArg(a++,buffer_from_tensor(idx));kernel.setArg(a++,cl_ulong(rows));kernel.setArg(a++,cl_ulong(axis));kernel.setArg(a++,cl_ulong(inner));kernel.setArg(a++,cl_ulong(k));kernel.setArg(a++,cl_int(largest));q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(rows),cl::NullRange,q.events(),q.event("topk_select"));sync_if_needed(self.device());}
 values.resize_(shape).copy_(v);indices.resize_(shape).copy_(idx);return {values,indices};
}
std::tuple<Tensor,Tensor,Tensor> unique_gpu(const Tensor &self,bool sorted,bool inverse,bool counts) {
 GUARD;
 // Target image IDs are small lists. Rank sorting is deliberately bounded;
 // a general large-array parallel sort is a separate optimization.
 if((self.scalar_type()!=kFloat && self.scalar_type()!=kLong && self.scalar_type()!=kInt) || self.numel()>4096){TORCH_WARN_ONCE("OpenCL unique: large input or unsupported dtype uses CPU");auto r=at::_unique2(self.cpu(),sorted,inverse,counts);return {std::get<0>(r).to(self.device()),std::get<1>(r).to(self.device()),std::get<2>(r).to(self.device())};}
 auto x=self.contiguous();auto n=x.numel();auto opts=self.options().dtype(kLong);
 auto inv=at::empty(inverse?self.sizes():IntArrayRef({0}),opts);
 if(!n)return {at::empty({0},self.options()),inv,at::empty({0},opts)};
 auto sx=at::empty({n},self.options()),order=at::empty({n},opts),u=at::empty({n},self.options()),c=at::empty({n},opts),total=at::empty({1},opts);
 auto q=getExecutionContext(self);dlprim::Context ctx(q);auto type=dlprim::data_type_to_opencl_type(todp(self.dtype()));std::ostringstream s;
 s<<"__kernel void rank_sort(__global const "<<type<<"*x,ulong xo,ulong n,__global "<<type<<"*s,__global long*o){ulong g=get_global_id(0);if(g>=n)return;"<<type<<" a=x[xo+g];ulong rank=0;for(ulong j=0;j<n;j++){"<<type<<" b=x[xo+j];int less=b<a,eq=b==a;";
 if(self.scalar_type()==kFloat)s<<"less=(!isnan(b)&&isnan(a)) || b<a;eq=eq || (isnan(b)&&isnan(a));";
 s<<"rank+=less||(eq&&j<g);}s[rank]=a;o[rank]=g;}"
 <<"__kernel void compact(__global const "<<type<<"*s,__global const long*o,ulong n,__global "<<type<<"*u,__global long*c,__global long*inv,ulong io,int want_inv,__global long*total){long count=0;for(ulong j=0;j<n;j++){if(j==0 || s[j]!=s[j-1]){u[count]=s[j];c[count]=0;count++;}c[count-1]++;if(want_inv)inv[io+o[j]]=count-1;}total[0]=count;}";
 auto p=program_for(ctx,s.str());cl::Kernel rank(p,"rank_sort");rank.setArg(0,buffer_from_tensor(x));rank.setArg(1,cl_ulong(x.storage_offset()));rank.setArg(2,cl_ulong(n));rank.setArg(3,buffer_from_tensor(sx));rank.setArg(4,buffer_from_tensor(order));q.queue().enqueueNDRangeKernel(rank,cl::NullRange,cl::NDRange(n),cl::NullRange,q.events(),q.event("unique_rank"));
 cl::Kernel compact(p,"compact");compact.setArg(0,buffer_from_tensor(sx));compact.setArg(1,buffer_from_tensor(order));compact.setArg(2,cl_ulong(n));compact.setArg(3,buffer_from_tensor(u));compact.setArg(4,buffer_from_tensor(c));compact.setArg(5,buffer_from_tensor(inverse?inv:order));compact.setArg(6,cl_ulong(0));compact.setArg(7,cl_int(inverse));compact.setArg(8,buffer_from_tensor(total));q.queue().enqueueNDRangeKernel(compact,cl::NullRange,cl::NDRange(1),cl::NDRange(1),q.events(),q.event("unique_compact"));
 auto count=total.item<int64_t>();return {u.narrow(0,0,count).clone(),inv,counts?c.narrow(0,0,count).clone():at::empty({0},opts)};
}
Tensor advanced_index(const Tensor &,const c10::List<c10::optional<Tensor>>&);
Tensor & index_put_gpu(Tensor &self,const c10::List<c10::optional<Tensor>> &indices,const Tensor &values,bool accumulate,bool unsafe) {
 GUARD;
 TORCH_CHECK(self.scalar_type()==values.scalar_type(),"index_put dtype mismatch");
 at::assert_no_internal_overlap(self);at::assert_no_overlap(self,values);
 if((self.scalar_type()!=kFloat && self.scalar_type()!=kLong && self.scalar_type()!=kInt && self.scalar_type()!=kBool && self.scalar_type()!=kByte && self.scalar_type()!=kChar) || self.dim()>8){
  TORCH_WARN_ONCE("OpenCL index_put: unsupported dtype/rank uses CPU");auto cpu=self.cpu();c10::List<c10::optional<Tensor>> ids;for(size_t j=0;j<indices.size();j++){auto i=indices.get(j);ids.push_back(i && i->defined()?c10::optional<Tensor>(i->cpu()):c10::nullopt);}at::_index_put_impl_(cpu,ids,values.cpu(),accumulate,unsafe);self.copy_(cpu);return self;}
 // Reuse the checked advanced-index planner to select physical addresses.
 // Addresses are generated on-device, including noncontiguous storage strides.
 auto addresses=at::empty(self.sizes(),self.options().dtype(kLong));
 auto q=getExecutionContext(self);dlprim::Context ctx(q);std::ostringstream s;
 s<<"__kernel void addresses(__global long*p,ulong n,ulong offset";for(int d=0;d<self.dim();d++)s<<",ulong s"<<d<<",ulong t"<<d;
 s<<"){ulong g=get_global_id(0);if(g>=n)return;ulong rem=g,a=offset;";for(int d=self.dim()-1;d>=0;d--)s<<"a+=(rem%s"<<d<<")*t"<<d<<";rem/=s"<<d<<";";s<<"p[g]=a;}";
 auto type=dlprim::data_type_to_opencl_type(todp(self.dtype()));
 s<<"__kernel void put(__global "<<type<<"*x,__global const long*p,ulong po,__global const "<<type<<"*v,ulong vo,ulong n,int accumulate){ulong g=get_global_id(0);if(g>=n)return;long address=p[po+g];"<<type<<" a=v[vo+g];if(!accumulate){x[address]=a;return;}";
 if(self.scalar_type()==kFloat)s<<"volatile __global unsigned int*dst=(volatile __global unsigned int*)(x+address);unsigned int old=atomic_cmpxchg(dst,0u,0u),seen;do{seen=old;old=atomic_cmpxchg(dst,seen,as_uint(as_float(seen)+a));}while(old!=seen);";
 else if(self.scalar_type()==kInt)s<<"atomic_add((volatile __global int*)(x+address),a);";
 else s<<"}"; // Small integers/long use destination-owned reduction below, avoiding optional int64 atomics.
 if(self.scalar_type()==kFloat || self.scalar_type()==kInt)s<<"}";
 else {
  s<<"__kernel void reduce_put(__global "<<type<<"*x,__global const long*all,ulong xn,__global const long*p,ulong po,__global const "<<type<<"*v,ulong vo,ulong n,int accumulate){ulong g=get_global_id(0);if(g>=xn)return;long address=all[g];"<<type<<" a=x[address];for(ulong j=0;j<n;j++)if(p[po+j]==address){if(accumulate)a="<<(self.scalar_type()==kBool?"(a||v[vo+j])":"a+v[vo+j]")<<";else a=v[vo+j];}x[address]=a;}";
 }
 auto program=program_for(ctx,s.str());
 if(self.numel()){cl::Kernel kernel(program,"addresses");int a=0;kernel.setArg(a++,buffer_from_tensor(addresses));kernel.setArg(a++,cl_ulong(self.numel()));kernel.setArg(a++,cl_ulong(self.storage_offset()));for(int d=0;d<self.dim();d++){kernel.setArg(a++,cl_ulong(self.size(d)));kernel.setArg(a++,cl_ulong(self.stride(d)));}q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(self.numel()),cl::NullRange,q.events(),q.event("index_put_addresses"));}
 auto selected=advanced_index(addresses,indices);auto v=values.to(self.device()).expand(selected.sizes()).contiguous();
 if(!selected.numel())return self;
 if(self.scalar_type()==kFloat || self.scalar_type()==kInt || !accumulate){cl::Kernel kernel(program,"put");int a=0;kernel.setArg(a++,buffer_from_tensor(self));kernel.setArg(a++,buffer_from_tensor(selected));kernel.setArg(a++,cl_ulong(selected.storage_offset()));kernel.setArg(a++,buffer_from_tensor(v));kernel.setArg(a++,cl_ulong(v.storage_offset()));kernel.setArg(a++,cl_ulong(selected.numel()));kernel.setArg(a++,cl_int(accumulate));q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(selected.numel()),cl::NullRange,q.events(),q.event("index_put_scatter"));}
 else {cl::Kernel kernel(program,"reduce_put");int a=0;kernel.setArg(a++,buffer_from_tensor(self));kernel.setArg(a++,buffer_from_tensor(addresses));kernel.setArg(a++,cl_ulong(self.numel()));kernel.setArg(a++,buffer_from_tensor(selected));kernel.setArg(a++,cl_ulong(selected.storage_offset()));kernel.setArg(a++,buffer_from_tensor(v));kernel.setArg(a++,cl_ulong(v.storage_offset()));kernel.setArg(a++,cl_ulong(selected.numel()));kernel.setArg(a++,cl_int(accumulate));q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(self.numel()),cl::NullRange,q.events(),q.event("index_put_reduce"));}
 sync_if_needed(self.device());return self;
}
std::tuple<Tensor&,Tensor&> pool_gpu(const Tensor &self,IntArrayRef kernel,IntArrayRef stride,IntArrayRef padding,IntArrayRef dilation,bool ceil_mode,Tensor &out,Tensor &indices) {
 GUARD;
 TORCH_CHECK(self.dim()==3 || self.dim()==4,"max_pool2d expects CHW or NCHW input");
 auto pair=[](IntArrayRef x,int fallback){TORCH_CHECK(x.size()<=2,"pool parameter expects one or two values");return std::pair<int64_t,int64_t>(x.empty()?fallback:x[0],x.empty()?fallback:x[x.size()==1?0:1]);};
 TORCH_CHECK(kernel.size()==1 || kernel.size()==2,"pool kernel expects one or two values");
 auto k=pair(kernel,0),s=stride.empty()?k:pair(stride,0),p=pair(padding,0),d=pair(dilation,1);
 TORCH_CHECK(k.first>0 && k.second>0 && s.first>0 && s.second>0 && d.first>0 && d.second>0 && p.first>=0 && p.second>=0 && p.first<=k.first/2 && p.second<=k.second/2,"invalid pooling parameters");
 TORCH_CHECK(out.device()==self.device() && out.scalar_type()==self.scalar_type() && indices.device()==self.device() && indices.scalar_type()==kLong,"pool output mismatch");
 if(self.scalar_type()!=kFloat){TORCH_WARN_ONCE("OpenCL max_pool2d_with_indices: unsupported dtype uses CPU");auto r=at::max_pool2d_with_indices(self.cpu(),kernel,stride,padding,dilation,ceil_mode);out.resize_(std::get<0>(r).sizes()).copy_(std::get<0>(r));indices.resize_(std::get<1>(r).sizes()).copy_(std::get<1>(r));return {out,indices};}
 int64_t h=self.size(-2),w=self.size(-1);TORCH_CHECK(h>0 && w>0 && self.size(-3)>0,"pool input spatial/channel dimensions must be nonempty");
 auto size=[ceil_mode](int64_t n,int64_t k,int64_t s,int64_t p,int64_t d){int64_t numerator=n+2*p-d*(k-1)-1+(ceil_mode?s-1:0);int64_t v=(numerator>=0?numerator/s:-((-numerator+s-1)/s))+1;if(ceil_mode && (v-1)*s>=n+p)--v;return v;};
 int64_t oh=size(h,k.first,s.first,p.first,d.first),ow=size(w,k.second,s.second,p.second,d.second);TORCH_CHECK(oh>0 && ow>0,"pool output size is too small");
 auto shape=self.sizes().vec();shape[shape.size()-2]=oh;shape.back()=ow;auto x=self.contiguous(),y=at::empty(shape,self.options()),idx=at::empty(shape,self.options().dtype(kLong));
 if(y.numel()){auto q=getExecutionContext(self);dlprim::Context ctx(q);const std::string code=R"CLC(
 __kernel void pool(__global const float*x,ulong xo,__global float*y,__global long*idx,ulong n,long h,long w,long oh,long ow,long kh,long kw,long sh,long sw,long ph,long pw,long dh,long dw){ulong g=get_global_id(0);if(g>=n)return;long plane=g/(oh*ow),oy=(g/ow)%oh,ox=g%ow,sy=oy*sh-ph,sx=ox*sw-pw;float best=-INFINITY;long bi=-1;for(long ky=0;ky<kh;ky++)for(long kx=0;kx<kw;kx++){long yy=sy+ky*dh,xx=sx+kx*dw;if(yy<0||yy>=h||xx<0||xx>=w)continue;long i=yy*w+xx;float a=x[xo+plane*h*w+i];if(bi<0||a>best||isnan(a)){best=a;bi=i;}}y[g]=best;idx[g]=bi;}
 )CLC";cl::Kernel op(program_for(ctx,code),"pool");int a=0;op.setArg(a++,buffer_from_tensor(x));op.setArg(a++,cl_ulong(x.storage_offset()));op.setArg(a++,buffer_from_tensor(y));op.setArg(a++,buffer_from_tensor(idx));op.setArg(a++,cl_ulong(y.numel()));for(auto z:{h,w,oh,ow,k.first,k.second,s.first,s.second,p.first,p.second,d.first,d.second})op.setArg(a++,cl_long(z));q.queue().enqueueNDRangeKernel(op,cl::NullRange,cl::NDRange(y.numel()),cl::NullRange,q.events(),q.event("max_pool_with_indices"));sync_if_needed(self.device());}
 out.resize_(shape).copy_(y);indices.resize_(shape).copy_(idx);return {out,indices};
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
 m.impl("aten::addmv.out",&ptdlprim::addmv_gpu);
 m.impl("aten::topk.values",&ptdlprim::topk_gpu);
 m.impl("aten::_unique2",&ptdlprim::unique_gpu);
 m.impl("aten::_index_put_impl_",&ptdlprim::index_put_gpu);
 m.impl("aten::max_pool2d_with_indices.out",&ptdlprim::pool_gpu);
}
