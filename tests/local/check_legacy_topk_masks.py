import torch,pytorch_ocl
torch.set_num_threads(2)
for dtype in [torch.float32,torch.int32]:
 x=torch.tensor([[[1]],[[1]]],dtype=dtype).expand(2,1,10)
 torch.testing.assert_close(x.to('ocl:0').bool().cpu(),x.bool())
for n in [1,5,65,1344]:
 x=torch.linspace(-1,1,n).repeat(2,1,1)
 g=x.to('ocl:0');k=min(n,10)
 v,i=g.topk(k,-1);cv,ci=x.topk(k,-1)
 torch.testing.assert_close(v.cpu(),cv);torch.testing.assert_close(i.cpu(),ci)
 print('PASS topk',n,flush=True)
x=torch.tensor([0.,.1,-.1,1.,float('nan')]).to('ocl:0')
torch.testing.assert_close(x.bool().cpu(),x.cpu().bool())
print('PASS expanded masks and boolean conversion',flush=True)
