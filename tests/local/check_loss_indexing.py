"""CPU-reference checks for nonzero and advanced indexing, including gradients."""
import warnings
import torch,pytorch_ocl

torch.manual_seed(67);torch.set_num_threads(2);torch.backends.mkldnn.enabled=False

def check(x,indices):
 def move(i):return i.to('ocl:0') if isinstance(i,torch.Tensor) else i
 g=x.to('ocl:0');gi=tuple(move(i) for i in indices)
 expected=x[indices];actual=g[gi]
 torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0,equal_nan=True)
 return actual

with warnings.catch_warnings(record=True) as caught:
 warnings.simplefilter('always')
 for shape in [(),(0,3),(1,),(127,),(128,),(129,),(513,),(5,7,13)]:
  for dtype in [torch.float32,torch.int64,torch.bool]:
   x=(torch.rand(shape)>.6).to(dtype)
   torch.testing.assert_close(torch.nonzero(x.to('ocl:0')).cpu(),torch.nonzero(x))
 x=torch.tensor([0.,-0.,float('nan'),float('inf'),-2.]);torch.testing.assert_close(x.to('ocl:0').nonzero().cpu(),x.nonzero())
 x=torch.arange(11*17.).reshape(11,17).t()[2:]
 torch.testing.assert_close(x.to('ocl:0').nonzero().cpu(),x.nonzero())
 x=torch.arange(4*5*6*7.).reshape(4,5,6,7)
 cases=[(torch.tensor([0,3,-1]),),
        (slice(None),torch.tensor([1,3])),
        (torch.tensor([[0],[2]]),slice(None),torch.tensor([[1,3,5]])),
        (slice(None),torch.tensor([[0],[3]]),torch.tensor([[1,4,5]])),
        (torch.tensor(2),),
        (torch.empty((0,),dtype=torch.long),),
        (torch.rand(4,5)>.5,),
        (slice(None),torch.rand(5,6)>.5),
        (torch.ones(4,5,6,dtype=torch.bool),),
        (torch.zeros(4,5,6,dtype=torch.bool),)]
 for i,idx in enumerate(cases):
  print('index case',i,flush=True);check(x,idx)
 check(x.permute(2,0,3,1),(torch.tensor([0,2]),slice(None),torch.tensor([1,4])))
 # Explicit CPU indices, negative indices and int32 indices on GPU self.
 torch.testing.assert_close(x.to('ocl:0')[torch.tensor([-1,1],dtype=torch.int32)].cpu(),x[torch.tensor([-1,1])])
 # Broadcast sources keep zero strides; no materialized expanded source needed.
 x=torch.randn(4,3,1,5).expand(4,3,1344,5);mask=torch.rand(4,3,1344)>.8
 check(x,(mask,))
 base=torch.randn(4,3,1,5);g=base.to('ocl:0').expand(4,3,1344,5)
 torch.testing.assert_close(g[mask.to('ocl:0')].cpu(),base.expand(4,3,1344,5)[mask])
 base=torch.arange(100.).reshape(10,10);g=base.to('ocl:0')[2:,1:]
 idx=torch.tensor([0,3,-1])
 torch.testing.assert_close(g[idx.to('ocl:0')].cpu(),base[2:,1:][idx])
 # Out variants and offset storage.
 x=torch.arange(80.).reshape(8,10)[2:];g=x.to('ocl:0')
 mask=(x%3==0);ref=x[mask]
 out=torch.empty(len(ref),2,device='ocl:0')[:,0]
 torch.ops.aten.index.Tensor_out(g,[mask.to('ocl:0')],out=out)
 torch.testing.assert_close(out.cpu(),ref)
 ref=x.nonzero();out=torch.empty(ref.shape[1],ref.shape[0],device='ocl:0',dtype=torch.long).t()
 torch.nonzero(g,out=out);torch.testing.assert_close(out.cpu(),ref)
 # Autograd continues through native index backward (CPU index_put fallback).
 for indices in [(torch.tensor([1,1,3]),),(torch.rand(5,7)>.5,),
                 (torch.tensor([0,3]),slice(None),torch.tensor([2,4]))]:
  shape=(5,7) if len(indices)<3 else (5,7,6)
  c=torch.randn(shape,requires_grad=True);g=c.detach().to('ocl:0').requires_grad_()
  moved=tuple(i.to('ocl:0') if isinstance(i,torch.Tensor) else i for i in indices)
  c[indices].square().sum().backward();g[moved].square().sum().backward()
  torch.testing.assert_close(g.grad.cpu(),c.grad)
 for idx in [torch.tensor([4]),torch.tensor([-5])]:
  try:torch.arange(4.,device='ocl:0')[idx.to('ocl:0')]
  except (RuntimeError,IndexError):pass
  else:raise AssertionError('out of bounds index did not raise')
 try:torch.ones(4,5,device='ocl:0')[torch.ones(3,dtype=torch.bool,device='ocl:0')]
 except (RuntimeError,IndexError):pass
 else:raise AssertionError('bad boolean shape did not raise')
 bad=[str(w.message) for w in caught if 'not currently supported' in str(w.message) and 'aten::_index_put_impl_' not in str(w.message)]
 assert not bad,bad
print('PASS: nonzero/index ordering, broadcasts/strides/masks, CPU indices, empty/scalar/out/error cases, index gradients')
