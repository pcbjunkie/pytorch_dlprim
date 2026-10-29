#include "utils.h"
#include <ATen/MemoryOverlap.h>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>
#include <iostream>
#include <limits>
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
 // Use explicit 32-bit metadata and word-pair int64 outputs on r600.
 bool legacy=ctx.device().getInfo<CL_DEVICE_ADDRESS_BITS>()==32;
 if(legacy && k<=32 && self.scalar_type()==kFloat) {
  TORCH_CHECK(self.numel()<=std::numeric_limits<cl_uint>::max() && x.storage_offset()<=std::numeric_limits<cl_uint>::max(),"topk exceeds uint32 addressing");
  bool little=ctx.device().getInfo<CL_DEVICE_ENDIAN_LITTLE>();
  s<<"__kernel void select_k(__global const float*x,ulong xo,__global float*v,__global uint2*i,ulong rows,ulong axis,ulong inner,ulong k,int largest){uint r=get_global_id(0);if(r>=rows)return;uint base=(r/(uint)inner)*(uint)axis*(uint)inner+r%(uint)inner,ob=(r/(uint)inner)*(uint)k*(uint)inner+r%(uint)inner;uint selected[32];for(uint t=0;t<(uint)k;t++){uint best=0;int found=0;float bv=0;for(uint j=0;j<(uint)axis;j++){int used=0;for(uint p=0;p<t;p++)used|=selected[p]==j;if(used)continue;float a=x[(uint)xo+base+j*(uint)inner];int better=!found || (isnan(a)?(largest&&!isnan(bv)):(isnan(bv)?!largest:(largest?a>bv:a<bv)));if(better){best=j;bv=a;found=1;}}selected[t]=best;v[ob+t*(uint)inner]=bv;i[ob+t*(uint)inner]=(uint2)("<<(little?"best,0":"0,best")<<");}}";
 } else {
 s<<"__kernel void select_k(__global const "<<type<<"*x,ulong xo,__global "<<type<<"*v,__global long*i,ulong rows,ulong axis,ulong inner,ulong k,int largest){ulong r=get_global_id(0);if(r>=rows)return;ulong base=(r/inner)*axis*inner+r%inner,ob=(r/inner)*k*inner+r%inner;for(ulong t=0;t<k;t++){long best=-1;"<<type<<" bv=0;for(ulong j=0;j<axis;j++){int used=0;for(ulong p=0;p<t;p++)used|=i[ob+p*inner]==(long)j;if(used)continue;"<<type<<" a=x[xo+base+j*inner];int better=best<0 || (largest?a>bv:a<bv);";
 if(self.scalar_type()==kFloat)s<<"better=best<0 || (isnan(a)?(largest&&!isnan(bv)):(isnan(bv)?!largest:(largest?a>bv:a<bv)));";
 s<<"if(better){best=j;bv=a;}}v[ob+t*inner]=bv;i[ob+t*inner]=best;}}";
 }
 cl::Kernel kernel(program_for(ctx,s.str()),"select_k");int a=0;kernel.setArg(a++,buffer_from_tensor(x));kernel.setArg(a++,cl_ulong(x.storage_offset()));kernel.setArg(a++,buffer_from_tensor(v));kernel.setArg(a++,buffer_from_tensor(idx));kernel.setArg(a++,cl_ulong(rows));kernel.setArg(a++,cl_ulong(axis));kernel.setArg(a++,cl_ulong(inner));kernel.setArg(a++,cl_ulong(k));kernel.setArg(a++,cl_int(largest));q.queue().enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(rows),cl::NullRange,q.events(),q.event("topk_select"));sync_if_needed(self.device());}
 values.resize_(shape).copy_(v);indices.resize_(shape).copy_(idx);return {values,indices};
}
std::tuple<Tensor,Tensor,Tensor> unique_gpu_legacy32(const Tensor &self,bool sorted,bool inverse,bool counts) {
 GUARD;
 // Target image IDs are small lists. Rank sorting is deliberately bounded;
 // a general large-array parallel sort is a separate optimization.
 if((self.scalar_type()!=kFloat && self.scalar_type()!=kLong && self.scalar_type()!=kInt) || self.numel()>4096){TORCH_WARN_ONCE("OpenCL unique: large input or unsupported dtype uses CPU");auto r=at::_unique2(self.cpu(),sorted,inverse,counts);return {std::get<0>(r).to(self.device()),std::get<1>(r).to(self.device()),std::get<2>(r).to(self.device())};}
 auto x=self.contiguous();auto n=x.numel();auto opts=self.options().dtype(kLong);
 auto inv=at::empty(inverse?self.sizes():IntArrayRef({0}),opts);
 if(!n)return {at::empty({0},self.options()),inv,at::empty({0},opts)};
 // The native path is bounded to 4096 elements: all work indices and counts
 // fit uint32. Keep ATen's int64 outputs as explicit pairs of 32-bit words;
 // Avoid serial int64 read/modify/write compaction on legacy runtimes.
 auto sx=at::empty({n},self.options()),order=at::empty({n},self.options().dtype(kInt));
 auto u=at::empty({n},self.options()),c=at::empty({n},opts),total=at::empty({1},self.options().dtype(kInt));
 auto q=getExecutionContext(self);dlprim::Context ctx(q);
 bool wide=self.scalar_type()==kLong;
 int lo=ctx.device().getInfo<CL_DEVICE_ENDIAN_LITTLE>() ? 0 : 1,hi=1-lo;
 std::string type=wide ? "uint2" : dlprim::data_type_to_opencl_type(todp(self.dtype()));
 std::ostringstream s;
 s<<"typedef "<<type<<" value_t;\n";
 s<<"inline uint2 pack_index(uint v){uint2 r=(uint2)(0,0);r.s"<<lo<<"=v;return r;}\n";
 if(wide) {
  s<<"inline int less_value(value_t a,value_t b){return as_int(a.s"<<hi<<")<as_int(b.s"<<hi<<") || (a.s"<<hi<<"==b.s"<<hi<<" && a.s"<<lo<<"<b.s"<<lo<<");}\n";
  s<<"inline int equal_value(value_t a,value_t b){return all(a==b);}\n";
 } else {
  s<<"inline int less_value(value_t a,value_t b){return ";
  if(self.scalar_type()==kFloat)s<<"(!isnan(a)&&isnan(b)) || ";
  s<<"a<b;}\ninline int equal_value(value_t a,value_t b){return a==b;}\n";
 }
 s<<"__kernel void rank_sort(__global const value_t*x,uint xo,uint n,__global value_t*s,__global uint*o){uint g=get_global_id(0);if(g>=n)return;value_t a=x[xo+g];uint rank=0;for(uint j=0;j<n;j++){value_t b=x[xo+j];int eq=equal_value(b,a);";
 if(self.scalar_type()==kFloat)s<<"eq=eq || (isnan(b)&&isnan(a));";
 s<<"rank+=less_value(b,a)||(eq&&j<g);}s[rank]=a;o[rank]=g;}\n"
   <<"__kernel void compact(__global const value_t*s,__global const uint*o,uint n,__global value_t*u,__global uint2*c,__global uint2*inv,int want_inv,__global uint*total){uint j=get_global_id(0);if(j>=n)return;uint count=1;for(uint k=1;k<=j;k++)count+=!equal_value(s[k],s[k-1]);if(j==0 || !equal_value(s[j],s[j-1])){u[count-1]=s[j];uint end=j+1;while(end<n && equal_value(s[end],s[j]))end++;c[count-1]=pack_index(end-j);}if(want_inv)inv[o[j]]=pack_index(count-1);if(j==n-1)total[0]=count;}";
 TORCH_CHECK(x.storage_offset()<=std::numeric_limits<cl_uint>::max(),"OpenCL unique input offset exceeds uint32 range");
 auto p=program_for(ctx,s.str());cl::Kernel rank(p,"rank_sort");rank.setArg(0,buffer_from_tensor(x));rank.setArg(1,cl_uint(x.storage_offset()));rank.setArg(2,cl_uint(n));rank.setArg(3,buffer_from_tensor(sx));rank.setArg(4,buffer_from_tensor(order));q.queue().enqueueNDRangeKernel(rank,cl::NullRange,cl::NDRange(n),cl::NullRange,q.events(),q.event("unique_rank"));
 sync_if_needed(self.device());
 cl::Kernel compact(p,"compact");compact.setArg(0,buffer_from_tensor(sx));compact.setArg(1,buffer_from_tensor(order));compact.setArg(2,cl_uint(n));compact.setArg(3,buffer_from_tensor(u));compact.setArg(4,buffer_from_tensor(c));compact.setArg(5,buffer_from_tensor(inverse?inv:c));compact.setArg(6,cl_int(inverse));compact.setArg(7,buffer_from_tensor(total));q.queue().enqueueNDRangeKernel(compact,cl::NullRange,cl::NDRange(n),cl::NullRange,q.events(),q.event("unique_compact"));
 sync_if_needed(self.device());
 auto count=total.item<int32_t>();
 TORCH_CHECK(count>=1 && count<=n,"OpenCL unique returned invalid count ",count," for ",n," elements");
 auto values_out=u.narrow(0,0,count).clone();
 auto counts_out=counts?c.narrow(0,0,count).clone():at::empty({0},opts);
 // Keep staging tensors alive until output copies complete in synchronous mode.
 sync_if_needed(self.device());
 return {values_out,inv,counts_out};
}
std::tuple<Tensor,Tensor,Tensor> unique_gpu(const Tensor &self,bool sorted,bool inverse,bool counts) {
 GUARD;
 auto select_queue=getExecutionContext(self);dlprim::Context select_context(select_queue);
 if(select_context.device().getInfo<CL_DEVICE_ADDRESS_BITS>()==32)
  return unique_gpu_legacy32(self,sorted,inverse,counts);
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
static Tensor &index_put_legacy32(Tensor &self,const c10::List<c10::optional<Tensor>> &indices,const Tensor &values,bool accumulate) {
 auto q=getExecutionContext(self);dlprim::Context ctx(q);
 auto fits=[](int64_t v){return v>=0 && uint64_t(v)<=std::numeric_limits<cl_uint>::max();};
 TORCH_CHECK(fits(self.numel()) && fits(self.storage_offset()),"legacy index_put exceeds uint32 addressing");
 for(int d=0;d<self.dim();++d)TORCH_CHECK(fits(self.size(d)) && fits(self.stride(d)),"legacy index_put layout exceeds uint32 addressing");
 int lo=ctx.device().getInfo<CL_DEVICE_ENDIAN_LITTLE>() ? 0 : 1,hi=1-lo;
 auto addresses=at::empty(self.sizes(),self.options().dtype(kLong));
 std::ostringstream code;
 code<<"inline __attribute__((always_inline)) uint2 put_divmod(uint n,uint d){uint q=0,r=0;for(int b=31;b>=0;--b){uint carry=r>>31;r=(r<<1)|((n>>b)&1u);if(carry || r>=d){r-=d;q|=1u<<b;}}return (uint2)(q,r);}";
 code<<"__kernel void addresses(__global uint2*p,uint n,uint offset";
 for(int d=0;d<self.dim();++d)code<<",uint s"<<d<<",uint t"<<d;
 code<<"){uint g=get_global_id(0);if(g>=n)return;uint rem=g,a=offset;";
 for(int d=self.dim()-1;d>=0;--d)code<<"uint2 qr"<<d<<"=put_divmod(rem,s"<<d<<");a+=qr"<<d<<".y*t"<<d<<";rem=qr"<<d<<".x;";
 code<<"uint2 result=(uint2)(0,0);result.s"<<lo<<"=a;p[g]=result;}";
 std::string type=self.scalar_type()==kLong ? "uint2" : dlprim::data_type_to_opencl_type(todp(self.dtype()));
 code<<"__kernel void put(__global "<<type<<"*x,__global const uint2*all,uint xn,__global const uint2*p,uint po,__global const "<<type<<"*v,uint vo,uint n,uint first){uint g=get_global_id(0);";
 if(!accumulate)code<<"if(g<n)x[p[po+g].s"<<lo<<"]=v[vo+g];}";
 else {
  // Destination ownership avoids floating-point CAS and repeated-index races.
  code<<"if(g>=xn)return;uint address=all[g].s"<<lo<<";"<<type<<" a=x[address];{uint end=first+min(n-first,64u);for(uint j=first;j<end;++j)if(p[po+j].s"<<lo<<"==address){";
  if(self.scalar_type()==kLong)code<<"uint2 b=v[vo+j];uint old=a.s"<<lo<<";a.s"<<lo<<"+=b.s"<<lo<<";a.s"<<hi<<"+=b.s"<<hi<<"+(a.s"<<lo<<"<old);";
  else if(self.scalar_type()==kInt)code<<"a=as_int(as_uint(a)+as_uint(v[vo+j]));";
  else if(self.scalar_type()==kBool)code<<"a=a||v[vo+j];";
  else code<<"a+=v[vo+j];";
  code<<"}}x[address]=a;}";
 }
 const char *sort_flag=std::getenv("DLPRIM_INDEX_SORT_REDUCE");
 bool sort_reduce=accumulate && self.scalar_type()==kFloat && sort_flag && std::strcmp(sort_flag,"1")==0;
 if(sort_reduce)code<<R"ocl(
 __kernel void index_sort_init32(__global const uint2 *p,uint po,uint n,
                                 __global uint *keys,__global uint *order) {
     uint g=get_global_id(0);
     keys[g]=g<n?p[po+g].s)ocl"<<lo<<R"ocl(:0xffffffffu;
     order[g]=g<n?g:0xffffffffu;
 }
 __kernel void index_sort_pass32(__global const uint *keys,__global const uint *order,
                                 __global uint *outkeys,__global uint *outorder,
                                 uint stage,uint step) {
     uint g=get_global_id(0),other=g^step;
     uint ak=keys[g],ai=order[g],bk=keys[other],bi=order[other];
     uint greater=(ak>bk)||((ak==bk)&&(ai>bi));
     uint less=(ak<bk)||((ak==bk)&&(ai<bi));
     uint want_min=((g&stage)==0)==((g&step)==0);
     uint take=want_min?greater:less;
     outkeys[g]=take?bk:ak;outorder[g]=take?bi:ai;
 }
 __kernel void index_segment_sum32(__global float *x,
                                   __global const uint *keys,__global const uint *order,
                                   __global const float *v,uint vo,uint n) {
     uint g=get_global_id(0);
     if(g>=n)return;
     uint key=keys[g];
     if(g>0){if(keys[g-1]==key)return;}
     // Stable secondary order preserves the original sequential addition order.
     float sum=x[key];
     for(uint j=g;j<n;++j){if(keys[j]!=key)break;sum+=v[vo+order[j]];}
     x[key]=sum;
 }
 )ocl";
 auto program=program_for(ctx,code.str());
 if(self.numel()) {
  cl::Kernel k(program,"addresses");int a=0;k.setArg(a++,buffer_from_tensor(addresses));k.setArg(a++,cl_uint(self.numel()));k.setArg(a++,cl_uint(self.storage_offset()));
  for(int d=0;d<self.dim();++d){k.setArg(a++,cl_uint(self.size(d)));k.setArg(a++,cl_uint(self.stride(d)));}
  q.queue().enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(self.numel()),cl::NullRange,q.events(),q.event("index_put_addresses32"));
  sync_if_needed(self.device());
 }
 auto selected=advanced_index(addresses,indices);
 auto v=values.to(self.device()).expand(selected.sizes()).contiguous();
 sync_if_needed(self.device());
 if(!selected.numel())return self;
 TORCH_CHECK(fits(selected.storage_offset()) && fits(selected.numel()) && fits(v.storage_offset()),"legacy index_put updates exceed uint32 addressing");
 if(sort_reduce && selected.numel()>=64 && selected.numel()<=(1<<20)) {
  size_t padded=32;while(padded<size_t(selected.numel()))padded*=2;
  Tensor keys[2],order[2];
  for(int i=0;i<2;++i){keys[i]=at::empty({int64_t(padded)},self.options().dtype(kInt));order[i]=at::empty({int64_t(padded)},self.options().dtype(kInt));}
  cl::Kernel init(program,"index_sort_init32");
  init.setArg(0,buffer_from_tensor(selected));init.setArg(1,cl_uint(selected.storage_offset()));init.setArg(2,cl_uint(selected.numel()));
  init.setArg(3,buffer_from_tensor(keys[0]));init.setArg(4,buffer_from_tensor(order[0]));
  q.queue().enqueueNDRangeKernel(init,cl::NullRange,cl::NDRange(padded),cl::NDRange(32),q.events(),q.event("index_sort_init32"));
  cl::Kernel pass(program,"index_sort_pass32");int src=0;
  for(size_t stage=2;stage<=padded;stage*=2)for(size_t step=stage/2;step;step/=2){
   pass.setArg(0,buffer_from_tensor(keys[src]));pass.setArg(1,buffer_from_tensor(order[src]));
   pass.setArg(2,buffer_from_tensor(keys[1-src]));pass.setArg(3,buffer_from_tensor(order[1-src]));
   pass.setArg(4,cl_uint(stage));pass.setArg(5,cl_uint(step));
   q.queue().enqueueNDRangeKernel(pass,cl::NullRange,cl::NDRange(padded),cl::NDRange(32),q.events(),q.event("index_sort_pass32"));src=1-src;
  }
  cl::Kernel reduce(program,"index_segment_sum32");
  reduce.setArg(0,buffer_from_tensor(self));reduce.setArg(1,buffer_from_tensor(keys[src]));reduce.setArg(2,buffer_from_tensor(order[src]));
  reduce.setArg(3,buffer_from_tensor(v));reduce.setArg(4,cl_uint(v.storage_offset()));reduce.setArg(5,cl_uint(selected.numel()));
  q.queue().enqueueNDRangeKernel(reduce,cl::NullRange,cl::NDRange(padded),cl::NDRange(32),q.events(),q.event("index_segment_sum32"));
  // Scratch and selected/value buffers must outlive all queued passes.
  q.queue().finish();return self;
 }
 cl::Kernel k(program,"put");int a=0;k.setArg(a++,buffer_from_tensor(self));k.setArg(a++,buffer_from_tensor(addresses));k.setArg(a++,cl_uint(self.numel()));k.setArg(a++,buffer_from_tensor(selected));k.setArg(a++,cl_uint(selected.storage_offset()));k.setArg(a++,buffer_from_tensor(v));k.setArg(a++,cl_uint(v.storage_offset()));k.setArg(a++,cl_uint(selected.numel()));
 // Bound both workgroup size and per-dispatch scan on legacy r600.
 // Launches execute in order and retain input ordering for repeated indices.
 size_t work=accumulate?self.numel():selected.numel();
 for(size_t first=0;first<size_t(selected.numel());first+=accumulate?64:size_t(selected.numel())) {
  k.setArg(a,cl_uint(first));
  q.queue().enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange((work+63)/64*64),cl::NDRange(64),q.events(),q.event(accumulate?"index_put_gather32":"index_put_scatter32"));
 }

 sync_if_needed(self.device());return self;
}
Tensor & index_put_gpu(Tensor &self,const c10::List<c10::optional<Tensor>> &indices,const Tensor &values,bool accumulate,bool unsafe) {
 GUARD;
 TORCH_CHECK(self.scalar_type()==values.scalar_type(),"index_put dtype mismatch");
 at::assert_no_internal_overlap(self);at::assert_no_overlap(self,values);
 if((self.scalar_type()!=kFloat && self.scalar_type()!=kLong && self.scalar_type()!=kInt && self.scalar_type()!=kBool && self.scalar_type()!=kByte && self.scalar_type()!=kChar) || self.dim()>8){
  TORCH_WARN_ONCE("OpenCL index_put: unsupported dtype/rank uses CPU");auto cpu=self.cpu();c10::List<c10::optional<Tensor>> ids;for(size_t j=0;j<indices.size();j++){auto i=indices.get(j);ids.push_back(i && i->defined()?c10::optional<Tensor>(i->cpu()):c10::nullopt);}at::_index_put_impl_(cpu,ids,values.cpu(),accumulate,unsafe);self.copy_(cpu);return self;}
 {auto q=getExecutionContext(self);dlprim::Context ctx(q);if(ctx.device().getInfo<CL_DEVICE_ADDRESS_BITS>()==32)return index_put_legacy32(self,indices,values,accumulate);}
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
