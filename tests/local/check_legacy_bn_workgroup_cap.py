"""CPU-reference BN module checks; only run on a fresh GPU boot.

Covers module output, running statistics, input/affine gradients and eval output.
It does not validate the separate native_batch_norm saved-invstd API convention.
"""
import os
assert os.environ.get('DLPRIM_BN_MAX_WG') in ('64','128'), 'enable the opt-in cap explicitly'
import copy
import torch
import pytorch_ocl
torch.set_num_threads(2)
torch.backends.mkldnn.enabled = False
torch.manual_seed(6791)
for shape in [(2,128,16,16),(2,3,9,11),(2,8,1,1),(2,16,32,32),(2,4,128,128)]:
    for affine in (True,False):
        cpu=torch.nn.BatchNorm2d(shape[1],eps=1e-3,momentum=.1,affine=affine)
        if affine:
            with torch.no_grad():
                cpu.weight.uniform_(.5,1.5);cpu.bias.uniform_(-.2,.2)
        gpu=copy.deepcopy(cpu).to('ocl:0')
        x=torch.randn(shape,requires_grad=True)
        gx=x.detach().to('ocl:0').requires_grad_()
        y=cpu(x);gy=gpu(gx)
        torch.testing.assert_close(gy.cpu(),y,rtol=4e-4,atol=5e-5)
        for attr in ('running_mean','running_var'):
            torch.testing.assert_close(getattr(gpu,attr).cpu(),getattr(cpu,attr),rtol=4e-4,atol=5e-5)
        dy=torch.randn_like(y);y.backward(dy);gy.backward(dy.to('ocl:0'))
        torch.testing.assert_close(gx.grad.cpu(),x.grad,rtol=2e-3,atol=2e-4)
        if affine:
            for name in ('weight','bias'):
                torch.testing.assert_close(getattr(gpu,name).grad.cpu(),getattr(cpu,name).grad,rtol=2e-3,atol=2e-4)
        cpu.eval();gpu.eval()
        with torch.no_grad():
            torch.testing.assert_close(gpu(gx).cpu(),cpu(x),rtol=4e-4,atol=5e-5)
        print('PASS batchnorm',shape,'affine',affine,'train/running stats/gradients/eval',flush=True)
print('PASS ten BN module cases, including two-stage statistics and gradient reductions',flush=True)
