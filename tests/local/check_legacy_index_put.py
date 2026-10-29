import torch,pytorch_ocl
torch.set_num_threads(2);torch.manual_seed(6970)
for dtype in (torch.float32,torch.int32,torch.int64,torch.bool,torch.uint8,torch.int8):
 for accum in (False,True):
  base=torch.zeros(6,8,dtype=dtype);gbase=base.to('ocl:0');c=base[1:5,2:7];g=gbase[1:5,2:7]
  ids=(torch.tensor([0,1,1,3]) if accum else torch.tensor([0,1,2,3]),torch.tensor([2,2,2,4]))
  v=torch.tensor([1,2,3,4],dtype=dtype);c.index_put_(ids,v,accumulate=accum);g.index_put_(tuple(i.to('ocl:0') for i in ids),v.to('ocl:0'),accumulate=accum)
  torch.testing.assert_close(gbase.cpu(),base,rtol=0,atol=0)
  mask=torch.tensor([[1,0,1,0,0],[0,1,0,0,1],[1,0,0,1,0],[0,0,1,0,1]],dtype=torch.bool)
  v=torch.tensor(1,dtype=dtype);c.index_put_((mask,),v,accumulate=accum);g.index_put_((mask.to('ocl:0'),),v.to('ocl:0'),accumulate=accum)
  torch.testing.assert_close(gbase.cpu(),base,rtol=0,atol=0)
  print('PASS',dtype,'accumulate',accum,'offset/strided, integer and mask updates',flush=True)
x=torch.zeros(5);g=x.to('ocl:0');idx=torch.arange(5200)%5;v=torch.ones(5200);x.index_put_((idx,),v,accumulate=True);g.index_put_((idx.to('ocl:0'),),v.to('ocl:0'),accumulate=True);torch.testing.assert_close(g.cpu(),x,rtol=0,atol=0);print('PASS 5200 repeated updates',flush=True)
x=torch.tensor([2**63-1,-2**63],dtype=torch.int64);g=x.to('ocl:0');idx=torch.tensor([0,1]);v=torch.tensor([2,-2],dtype=torch.int64);x.index_put_((idx,),v,accumulate=True);g.index_put_((idx.to('ocl:0'),),v.to('ocl:0'),accumulate=True);torch.testing.assert_close(g.cpu(),x);print('PASS int64 wraparound',flush=True)
