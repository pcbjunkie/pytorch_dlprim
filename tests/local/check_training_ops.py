"""GPU coverage of the operators reported by the full wildlife training log."""
import gc
import warnings
import torch
import pytorch_ocl

torch.set_num_threads(2)
torch.backends.mkldnn.enabled = False
torch.manual_seed(31)
def close(g,c,**kw): torch.testing.assert_close(g.cpu(),c,**kw)
with warnings.catch_warnings(record=True) as caught:
 warnings.simplefilter('always')
 print('addmv',flush=True)
 for m,n in [(7,11),(0,3),(4,0)]:
  mat=torch.randn(m,n);vec=torch.randn(n)
  for beta in [0.,1.,-.3]:
   for alpha in [0.,1.,.7]:
    bias=torch.full((m,),float('nan')) if beta==0 else torch.randn(m)
    out=torch.empty(m*2,device='ocl:0')[::2]
    torch.addmv(bias.to('ocl:0'),mat.to('ocl:0'),vec.to('ocl:0'),beta=beta,alpha=alpha,out=out)
    close(out,torch.addmv(bias,mat,vec,beta=beta,alpha=alpha),equal_nan=True)
 bias=torch.randn(7);mat=torch.randn(7,9);vec=torch.randn(9);out=bias.to('ocl:0')
 torch.addmv(out,mat.to('ocl:0'),vec.to('ocl:0'),out=out);close(out,torch.addmv(bias,mat,vec))
 print('topk',flush=True)
 for dtype in [torch.float32,torch.int32,torch.int64]:
  for dim in [0,1,2,-1]:
   x=(torch.randperm(3*5*7).reshape(3,5,7)-50).to(dtype).transpose(0,2)
   for k in [0,1,min(3,x.size(dim)),x.size(dim)]:
    for largest in [True,False]:
     v,i=torch.topk(x.to('ocl:0'),k,dim=dim,largest=largest)
     a,b=torch.topk(x,k,dim=dim,largest=largest);close(v,a);close(i,b)
 for largest in [True,False]:
  x=torch.tensor([float('nan'),3.,3.,float('inf'),-float('inf'),-2.,float('nan')])
  v,i=x.to('ocl:0').topk(7,largest=largest);a,b=x.topk(7,largest=largest)
  close(v,a,equal_nan=True);close(x[i.cpu()],v.cpu(),equal_nan=True);assert len(i.cpu().unique())==7
 for k in [0,1]:
  x=torch.tensor(3.);v,i=x.to('ocl:0').topk(k);close(v,x);close(i,torch.tensor(0))
 print('unique',flush=True)
 for dtype in [torch.float32,torch.int32,torch.int64]:
  for shape in [(0,),(3,7),(4,5,2),()]:
   x=torch.randint(-3,8,shape).to(dtype)
   for inv in [False,True]:
    for counts in [False,True]:
     actual=torch.ops.aten._unique2(x.to('ocl:0'),True,inv,counts)
     expected=torch.ops.aten._unique2(x,True,inv,counts)
     for a,b in zip(actual,expected):close(a,b)
 x=torch.tensor([float('nan'),0.,-0.,float('inf'),-float('inf'),float('nan'),2.,2.])
 for a,b in zip(torch.ops.aten._unique2(x.to('ocl:0'),True,True,True),torch.ops.aten._unique2(x,True,True,True)):close(a,b,equal_nan=True)
 print('index put',flush=True)
 for dtype in [torch.float32,torch.int32,torch.int64,torch.bool,torch.uint8]:
  for acc in [False,True]:
   c=torch.zeros(5,7,dtype=dtype);g=c.to('ocl:0');ids=(torch.tensor([0,1,1,4,-1]),torch.tensor([2,3,3,1,1]));v=torch.ones(5,dtype=dtype)
   c.index_put_(ids,v,accumulate=acc);g.index_put_(tuple(i.to('ocl:0') for i in ids),v.to('ocl:0'),accumulate=acc);close(g,c)
 for acc in [False,True]:
  c=torch.randn(5,7);g=c.to('ocl:0');mask=torch.rand(5,7)>.4;v=torch.randn(mask.sum())
  c.index_put_((mask,),v,accumulate=acc);g.index_put_((mask.to('ocl:0'),),v.to('ocl:0'),accumulate=acc);close(g,c)
 c=torch.randn(7,8);g=c.to('ocl:0');cv=c.t()[1:6,2:];gv=g.t()[1:6,2:];ids=torch.tensor([0,2,-1]);v=torch.randn(3,5)
 cv[ids]=v;gv[ids.to('ocl:0')]=v.to('ocl:0');close(g,c)
 c=torch.randn(4,5,6);g=c.to('ocl:0');ids=(torch.tensor([[0],[3]]),slice(None),torch.tensor([[1,3,5]]));v=torch.randn(2,3,5)
 c[ids]=v;g[tuple(i.to('ocl:0') if isinstance(i,torch.Tensor) else i for i in ids)]=v.to('ocl:0');close(g,c)
 c=torch.randn(5,7,requires_grad=True);g=c.detach().to('ocl:0').requires_grad_();i=torch.tensor([1,1,3]);c[i].square().sum().backward();g[i.to('ocl:0')].square().sum().backward();close(g.grad,c.grad)
 try:torch.zeros(3,device='ocl:0').index_put_((torch.tensor([4],device='ocl:0'),),torch.ones(1,device='ocl:0'))
 except (IndexError,RuntimeError):pass
 else:raise AssertionError('invalid index accepted')
 print('pool',flush=True)
 for shape in [(2,3,9,11),(3,9,11),(0,3,9,11)]:
  x=torch.randn(shape)
  for kernel,stride,pad,dilation,ceil in [(2,2,0,1,False),(3,2,1,1,True),(3,1,1,2,False),(5,1,2,1,False)]:
   a,b=torch.nn.functional.max_pool2d(x.to('ocl:0'),kernel,stride,pad,dilation,ceil,return_indices=True)
   c,d=torch.nn.functional.max_pool2d(x,kernel,stride,pad,dilation,ceil,return_indices=True);close(a,c);close(b,d)
 x=torch.randn(2,3,9,11).transpose(-1,-2)
 expected=torch.nn.functional.max_pool2d(x,3,2,1,return_indices=True)
 out=torch.empty(*expected[0].shape,2,device='ocl:0')[...,0]
 idx=torch.empty(*expected[1].shape,2,device='ocl:0',dtype=torch.long)[...,0]
 torch.ops.aten.max_pool2d_with_indices.out(x.to('ocl:0'),[3,3],[2,2],[1,1],[1,1],False,out=out,indices=idx)
 close(out,expected[0]);close(idx,expected[1])
 x=torch.tensor([[[float('nan'),-float('inf')],[float('nan'),-float('inf')]]]);a,b=torch.nn.functional.max_pool2d(x.to('ocl:0'),2,return_indices=True);c,d=torch.nn.functional.max_pool2d(x,2,return_indices=True);close(a,c,equal_nan=True);close(b,d)
 bad=[str(w.message) for w in caught if 'CPU' in str(w.message) or 'not currently supported' in str(w.message)]
 assert not bad,bad
print('allocator counters',flush=True)
gc.collect();torch.ocl.empty_cache();base=torch.ocl.memory_stats();torch.ocl.reset_peak_memory_stats()
x=torch.empty(1003,device='ocl:0');view=x[1:];stats=torch.ocl.memory_stats();assert stats['allocated_bytes.all.current']-base['allocated_bytes.all.current']==4012
assert stats['reserved_bytes.all.current']>=stats['allocated_bytes.all.current'];del x;assert torch.ocl.memory_allocated()==stats['allocated_bytes.all.current'];del view;gc.collect()
assert torch.ocl.memory_allocated()==base['allocated_bytes.all.current'];assert torch.ocl.memory_reserved()>=stats['reserved_bytes.all.current']
x=torch.empty(1003,device='ocl:0');assert torch.ocl.memory_reserved()==stats['reserved_bytes.all.current'];torch.ocl.empty_cache();assert torch.ocl.memory_allocated()==base['allocated_bytes.all.current']+4012
assert torch.ocl.memory_reserved()>=torch.ocl.memory_allocated();del x;gc.collect();torch.ocl.empty_cache();assert torch.ocl.memory_reserved()==base['reserved_bytes.all.current']
torch.ocl.reset_peak_memory_stats();assert torch.ocl.max_memory_allocated()==torch.ocl.memory_allocated();assert torch.ocl.max_memory_reserved()==torch.ocl.memory_reserved()
print('PASS: GPU training operators and allocator counters',flush=True)
