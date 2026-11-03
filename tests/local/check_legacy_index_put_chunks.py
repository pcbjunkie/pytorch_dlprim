"""Repeated-index accumulation across dispatch boundaries, against CPU."""
import torch,pytorch_ocl
torch.set_num_threads(2);torch.manual_seed(65)
for n in [1,63,64,65,127,128,129,1280]:
 x=torch.randn(257)
 idx=torch.randint(0,257,(n,));v=torch.randn(n)
 ref=x.clone().index_put_((idx,),v,accumulate=True)
 g=x.to('ocl:0');g.index_put_((idx.to('ocl:0'),),v.to('ocl:0'),accumulate=True)
 torch.testing.assert_close(g.cpu(),ref,rtol=1e-5,atol=2e-6)
 print('PASS repeated updates',n,flush=True)
x=torch.zeros(2,1344,64);idx=torch.arange(20)*51
v=torch.randn(2,20,64)
ref=x.clone();ref.index_put_((torch.arange(2)[:,None],idx[None,:]),v,accumulate=True)
g=x.to('ocl:0');g.index_put_((torch.arange(2).to('ocl:0')[:,None],idx.to('ocl:0')[None,:]),v.to('ocl:0'),accumulate=True)
torch.testing.assert_close(g.cpu(),ref,rtol=1e-5,atol=2e-6)
print('PASS large sparse gradient layout',flush=True)
