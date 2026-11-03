"""Regression for dynamic-local score/index storage aliasing on radeonsi Clover."""
import torch
import pytorch_ocl

torch.set_num_threads(2)
torch.manual_seed(943)
checks=0
for rows in (1,2,7):
    source=torch.rand(rows,1,1344)
    source[source<.7]=0
    gpu=source.to('ocl:0')
    for largest in (False,True):
        for k in (1,10,32):
            expected=source.topk(k,largest=largest).values
            for repeat in range(3):
                values,indices=gpu.topk(k,largest=largest)
                values,indices=values.cpu(),indices.cpu()
                torch.testing.assert_close(values,expected,rtol=0,atol=0)
                torch.testing.assert_close(source.gather(-1,indices),values,rtol=0,atol=0)
                assert not bool((indices.sort().values.diff(dim=-1)==0).any())
                checks+=1
print('PASS',checks,'top-k local-storage comparisons on',torch.ocl.get_device_name(0))
