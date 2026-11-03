"""CPU parity and dispatch coverage for depthwise filter-gradient selection.

Run with DLPRIM_FILTER_SPLIT_K=1 and the candidate OpenCL backend. An output
directory argument keeps profiling artifacts separate from source/tests.
"""
import argparse
from pathlib import Path

import torch
import pytorch_ocl
from torch.nn import functional as F


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--only-case',type=int)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.mkldnn.enabled=False
    torch.backends.nnpack.set_flags(False)
    torch.manual_seed(2920)
    torch.ocl.enable_profiling(0)
    # batch, CI, CO, H, W, kernel, stride, groups, dilation
    cases=[(20,64,64,32,32,3,1,64,1),(20,128,128,16,16,3,1,128,1),
        (20,256,256,8,8,3,1,256,1),(2,8,8,128,128,3,1,8,1),
        (20,8,8,64,64,3,1,8,1),(2,4,4,17,19,3,1,4,1),
        (2,4,4,17,19,1,1,4,1),(2,4,4,17,19,5,1,4,1),
        (2,4,8,32,32,3,1,4,1),(2,8,12,32,32,3,1,4,1),
        (2,4,4,32,32,3,2,4,1),(2,4,4,32,32,3,1,4,2)]

    def shifted(value,offset):
        base=torch.empty(value.numel()+offset+17,device='ocl:0')
        view=base[offset:offset+value.numel()].view(value.shape)
        view.copy_(value)
        assert view.storage_offset()==offset
        return view

    for index,(batch,ci,co,h,w,k,stride,groups,dilation)in enumerate(cases):
        if args.only_case is not None and index!=args.only_case:
            continue
        x=torch.randn(batch,ci,h,w)
        weight=torch.randn(co,ci//groups,k,k,requires_grad=True)
        pad=(k//2)*dilation
        output=F.conv2d(x,weight,padding=pad,stride=stride,groups=groups,dilation=dilation)
        dy=torch.randn_like(output);output.backward(dy);expected=weight.grad
        gx,gw,gdy=shifted(x,7),shifted(weight.detach(),11),shifted(dy,13)
        trace=args.output/f'case-{index}.csv'
        for repeat in range(3):
            with torch.ocl.profile(0,str(trace)if repeat==0 else None):
                actual=torch.ops.aten.convolution_backward.default(gdy,gx,gw,None,
                    [stride,stride],[pad,pad],[dilation,dilation],False,[0,0],groups,
                    [False,True,False])[1]
                actual_cpu=actual.cpu()
                try:
                    torch.testing.assert_close(actual_cpu,expected,rtol=3e-3,atol=3e-3)
                except AssertionError:
                    torch.save(dict(case=index,input=x,weight=weight.detach(),dy=dy,
                        expected=expected,actual=actual_cpu),args.output/'failure.pt')
                    raise
        specialized=(groups==ci==co and ci>1 and stride==1 and dilation==1)
        events=trace.read_text()
        assert ('sep_conv_bw_filter' in events)==specialized,(index,events)
        if specialized:
            assert 'conv_filter_partials'not in events
            # Public autograd must add a second backward result to the leaf.
            leaf=gw.detach().requires_grad_()
            for repeat in range(2):
                F.conv2d(gx,leaf,padding=pad,stride=stride,groups=groups).backward(gdy)
                torch.testing.assert_close(leaf.grad.cpu(),expected*(repeat+1),rtol=3e-3,atol=6e-3)
        print('PASS',index,(batch,ci,co,h,w,k,stride,groups,dilation),
            'specialized'if specialized else'generic fallback',flush=True)
    print('PASS depthwise filter selection, offsets, reductions and accumulation',flush=True)


if __name__=='__main__':main()
