"""CPU-reference regression checks for utilization-related backend additions."""
import warnings
import torch
import pytorch_ocl

torch.backends.mkldnn.enabled=False
torch.set_num_threads(2)
torch.manual_seed(19)

def compare(actual, expected, **kw):
    torch.testing.assert_close(actual.cpu(),expected,**kw)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    for shape in [(97,), (3,7,11), ()]:
        original=torch.randn(shape)
        inputs=[original]
        if len(shape)>1: inputs.append(original.transpose(0,2))
        for x in inputs:
            for dim in [None, (), (0,)] if x.ndim else [None]:
                for keep in [False,True]:
                    for p in [0,1,2,3.5,-2,float('inf'),-float('inf')]:
                        compare(torch.linalg.vector_norm(x.to('ocl:0'),p,dim,keep),torch.linalg.vector_norm(x,p,dim,keep),rtol=2e-5,atol=2e-6)
    for x in [torch.zeros(12),torch.tensor([1.,float('nan'),2.]),torch.tensor([float('inf'),3.]),torch.empty(0,3)]:
        for p in [0,1,2,float('inf'),-float('inf')]:
            if x.numel()==0 and p in (float('inf'),-float('inf')):continue
            compare(torch.linalg.vector_norm(x.to('ocl:0'),p,dim=0),torch.linalg.vector_norm(x,p,dim=0),equal_nan=True)
    x=torch.randn(4,5)
    out=torch.empty(5,2,device='ocl:0')[:,0]
    torch.linalg.vector_norm(x.to('ocl:0'),dim=0,out=out)
    compare(out,torch.linalg.vector_norm(x,dim=0))
    params=[torch.nn.Parameter(torch.randn(3,7)),torch.nn.Parameter(torch.randn(13))]
    gpu=[torch.nn.Parameter(p.detach().to('ocl:0')) for p in params]
    for c,g in zip(params,gpu):
        c.grad=torch.randn_like(c);g.grad=c.grad.to('ocl:0')
    compare(torch.nn.utils.clip_grad_norm_(gpu,.5),torch.nn.utils.clip_grad_norm_(params,.5),rtol=2e-5,atol=2e-6)
    for c,g in zip(params,gpu):compare(g.grad,c.grad,rtol=2e-5,atol=2e-6)
    for dtype in [torch.float32,torch.int64,torch.int8]:
        a=torch.arange(12).reshape(3,4).to(dtype).t()
        mask=torch.tensor([[True],[False],[True],[False]])
        compare(torch.where(mask.to('ocl:0'),a.to('ocl:0'),torch.tensor(2,dtype=dtype).to('ocl:0')),torch.where(mask,a,2))
        g=a.to('ocl:0');g.masked_fill_(mask.to('ocl:0'),-3)
        compare(g,a.masked_fill(mask,-3))
    for dtype in [torch.float32,torch.int8,torch.int64]:
        for dim in [0,1,-1]:
            x=torch.arange(24).reshape(4,6).to(dtype)
            idx=torch.tensor([[0,1,1],[1,1,0]])
            if dim==0:idx=idx.t()
            src=torch.arange(idx.numel()).reshape(idx.shape).to(dtype)
            compare(x.to('ocl:0').scatter_add(dim,idx.to('ocl:0'),src.to('ocl:0')),x.scatter_add(dim,idx,src))
            g=x.to('ocl:0');g.scatter_add_(dim,idx.to('ocl:0'),src.to('ocl:0'))
            compare(g,x.scatter_add(dim,idx,src))
            compare(x.to('ocl:0').scatter(dim,idx.to('ocl:0'),3),x.scatter(dim,idx,3))
    x=torch.randn(3,4).t();idx=torch.tensor([[0,0],[1,1],[0,1],[1,0]])
    src=torch.randn(2,4).t()
    compare(x.to('ocl:0').scatter_add(1,idx.to('ocl:0'),src.to('ocl:0')),x.scatter_add(1,idx,src))
    # Scalar and large-index cases intentionally retain the CPU path.
    for x,idx,src,dim in [(torch.tensor(2.),torch.tensor(0),torch.tensor(3.),0),
                           (torch.zeros(3,40),torch.arange(35).repeat(3,1),torch.ones(3,35),1)]:
        compare(x.to('ocl:0').scatter_add(dim,idx.to('ocl:0'),src.to('ocl:0')),x.scatter_add(dim,idx,src))
    x=torch.randn(4,3).t(); idx=torch.tensor([[0,1],[1,2],[0,0]]);src=torch.randn(3,2)
    out=torch.empty(4,3,device='ocl:0').t()
    torch.scatter_add(x.to('ocl:0'),1,idx.to('ocl:0'),src.to('ocl:0'),out=out)
    compare(out,x.scatter_add(1,idx,src))
    try:
        torch.zeros(3,device='ocl:0').scatter_add(0,torch.tensor([3],device='ocl:0'),torch.ones(1,device='ocl:0'))
    except RuntimeError:pass
    else:raise AssertionError('out-of-range scatter did not raise')
    unexpected=[str(w.message) for w in caught if 'not currently supported' in str(w.message)]
    assert not unexpected,unexpected
print('PASS: norms, clipping, where, masked_fill, scatter_add/scatter; no backend fallback warnings')
