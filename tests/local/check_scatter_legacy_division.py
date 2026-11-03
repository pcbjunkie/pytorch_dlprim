"""CPU parity for scatter coordinate decoding on non-power-of-two dimensions."""
import torch,pytorch_ocl

torch.set_num_threads(2)
torch.manual_seed(6850)
for dtype in [torch.float32,torch.int64,torch.int8]:
    for shape in [(1,1344,5),(2,1344,5),(24,1344,5),(1,1344,4),(1344,5),(1,32,5)]:
        for varying in [False,True]:
            indices=torch.randint(shape[-1],(*shape[:-1],1)) if varying else torch.zeros((*shape[:-1],1),dtype=torch.long)
            for repeat in range(2):
                expected=torch.zeros(shape,dtype=dtype).scatter_(-1,indices,1)
                actual=torch.zeros(shape,dtype=dtype,device='ocl:0').scatter_(-1,indices.to('ocl:0'),1)
                torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
        print('PASS one-hot',dtype,shape,flush=True)
    for dim in [0,1,2]:
        base=torch.randint(-4,5,(3,7,11),dtype=torch.int64).to(dtype)
        shape=[2,5,9];shape[dim]=3
        index=torch.randint(base.shape[dim],shape)
        src=torch.randint(-3,4,shape,dtype=torch.int64).to(dtype)
        expected=base.scatter_add(dim,index,src)
        actual=base.to('ocl:0').scatter_add(dim,index.to('ocl:0'),src.to('ocl:0'))
        torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
    parent=torch.zeros(4,17,13,dtype=dtype)
    gpu=parent.to('ocl:0')
    view=parent[1:3,:,1:10:3];gview=gpu[1:3,:,1:10:3]
    idx=torch.randint(3,(2,17,1))
    view.scatter_(2,idx,7);gview.scatter_(2,idx.to('ocl:0'),7)
    torch.testing.assert_close(gpu.cpu(),parent,rtol=0,atol=0)
    print('PASS repeated-index additions and offset/noncontiguous output guards',dtype,flush=True)
for bad in [-1,5,(1<<32)+1,-(1<<32)]:
    try:
        torch.zeros(1,7,5,device='ocl:0').scatter_(2,torch.full((1,7,1),bad,dtype=torch.long).to('ocl:0'),1)
    except (RuntimeError,IndexError):pass
    else:raise AssertionError('invalid scatter index accepted')
print('PASS scatter non-power-of-two dimensions, batch sizes, dtypes, strides, accumulation and bounds',flush=True)
