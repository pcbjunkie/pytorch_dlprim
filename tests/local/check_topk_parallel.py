"""Check the cooperative top-k selector, its layout handling and serial fallbacks."""
import argparse
import csv
import json
import math
from pathlib import Path
import tempfile
import warnings

import torch
import pytorch_ocl

torch.set_num_threads(2)
torch.backends.mkldnn.enabled = False
torch.backends.nnpack.set_flags(False)
torch.manual_seed(139)
torch.ocl.enable_profiling(0)
checks = 0


def check(c, g, k, dim=-1, largest=True, sorted=True, out=None):
    global checks
    actual = torch.topk(g, k, dim=dim, largest=largest, sorted=sorted, out=out)
    expected = torch.topk(c, k, dim=dim, largest=largest, sorted=True)
    values, indices = (t.cpu() for t in actual)
    # Returning a sorted answer for sorted=False is allowed.
    torch.testing.assert_close(values, expected.values, rtol=0, atol=0, equal_nan=True)
    if c.dim():
        torch.testing.assert_close(c.gather(dim, indices), values, rtol=0, atol=0, equal_nan=True)
        if k:
            assert int(indices.min()) >= 0 and int(indices.max()) < c.size(dim)
            assert bool((indices.sort(dim=dim).values.diff(dim=dim) != 0).all())
    else:
        assert indices.item() == 0
    checks += 1
    return values, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for shape in [(1, 64), (7, 65), (3, 67), (20, 3, 1344), (20, 5, 1344), (2, 4097)]:
            c = torch.randperm(math.prod(shape)).reshape(shape).float()
            g = c.to('ocl:0')
            for k in (1, 10, 32):
                for largest in (True, False):
                    for ordered in (True, False):
                        check(c, g, k, largest=largest, sorted=ordered)
        print('PASS unique values, irregular rows/axes and k=1/10/32', flush=True)

        # Selected values and indices must agree even when CPU's tie order
        # differs. Also preserve the pre-existing lowest-index tie policy.
        pattern = [float('nan'), 0., -0., 1., 1., -2., float('inf'), -float('inf')]
        c = torch.tensor((pattern * 18)[:139]).repeat(3, 1)
        g = c.to('ocl:0')
        for largest in (True, False):
            order = sorted(range(c.shape[-1]), key=lambda j: (
                -int(math.isnan(c[0, j])) if largest else int(math.isnan(c[0, j])),
                0. if math.isnan(c[0, j]) else (-float(c[0, j]) if largest else float(c[0, j])), j))
            for k in (1, 10, 32):
                _, indices = check(c, g, k, largest=largest)
                torch.testing.assert_close(indices, torch.tensor(order[:k]).repeat(3, 1), rtol=0, atol=0)
        for value in (0., -0., float('nan'), float('inf'), -float('inf')):
            c = torch.full((3, 97), value); g = c.to('ocl:0')
            for largest in (True, False):
                _, indices = check(c, g, 32, largest=largest)
                torch.testing.assert_close(indices, torch.arange(32).repeat(3, 1), rtol=0, atol=0)
        print('PASS tied values, signed zero, NaNs and infinities', flush=True)

        storage = torch.randperm(300).float()
        gs = storage.to('ocl:0')
        c = storage[17:17+3*67].reshape(3, 67)
        g = gs[17:17+3*67].reshape(3, 67)
        assert g.storage_offset() == 17
        for largest in (True, False):
            vp = torch.full((2*3*10+11,), -12345., device='ocl:0')
            ip = torch.full((2*3*10+11,), -12345, dtype=torch.long, device='ocl:0')
            vo = vp[5:5+2*3*10:2].reshape(3, 10)
            io = ip[5:5+2*3*10:2].reshape(3, 10)
            check(c, g, 10, largest=largest, out=(vo, io))
            for parent in (vp.cpu(), ip.cpu()):
                mask = torch.ones_like(parent, dtype=torch.bool)
                mask[5:5+2*3*10:2] = False
                assert bool((parent[mask] == -12345).all())
        c = torch.randperm(5*67*3).float().reshape(5, 67, 3).transpose(1, 2)
        g = c.to('ocl:0')
        assert not g.is_contiguous()
        check(c, g, 10)
        print('PASS storage offsets, noncontiguous inputs/outputs and output guards', flush=True)

        # These cases must continue to work through the existing selector.
        for dtype in (torch.float32, torch.int32, torch.int64):
            c = torch.randperm(3*67).reshape(3, 67).to(dtype); g = c.to('ocl:0')
            for k in (0, 1, 33, 67):
                for largest in (True, False):
                    check(c, g, k, largest=largest)
        # Word-pair comparisons must preserve signed int64 values beyond the
        # FP32 and int32 exact ranges, including non-last-axis addressing.
        c = (torch.randperm(3*67).reshape(3, 67).long()-100) * (1 << 42)
        c[:, ::3] += (1 << 31) + 17
        for dim in (0, 1):
            for largest in (True, False):
                check(c, c.to('ocl:0'), min(10, c.size(dim)), dim=dim, largest=largest)
        c = torch.randperm(2*67*3).reshape(2, 67, 3).float()
        for largest in (True, False):
            check(c, c.to('ocl:0'), 33, dim=1, largest=largest)
        c = torch.randperm(3*5*7).reshape(3, 5, 7).float(); g = c.to('ocl:0')
        for dim in (0, 1, 2, -1):
            check(c, g, min(3, c.size(dim)), dim=dim)
        for shape in ((0, 67), (2, 0)):
            c = torch.empty(shape); check(c, c.to('ocl:0'), 0)
        for k in (0, 1):
            c = torch.tensor(3.); check(c, c.to('ocl:0'), k)
        print('PASS k=0/33/full-axis, integer, non-last-axis, empty and scalar fallbacks', flush=True)

        c = torch.randperm(3*67).float().reshape(3, 67).requires_grad_()
        g = c.detach().to('ocl:0').requires_grad_()
        weights = torch.arange(1, 11).float()
        (c.topk(10).values * weights).sum().backward()
        # ATen top-k backward uses scatter.src_out, a separate existing CPU
        # fallback. Check the resulting gradient and record that limitation;
        # it is not a fallback in the forward selector under test.
        with warnings.catch_warnings(record=True) as backward_warnings:
            warnings.simplefilter('always')
            (g.topk(10).values * weights.to('ocl:0')).sum().backward()
        torch.testing.assert_close(g.grad.cpu(), c.grad, rtol=0, atol=0)
        backward_messages = [str(w.message) for w in backward_warnings]
        assert all('aten::scatter.src_out' in message for message in backward_messages), backward_messages
        print('PASS top-k autograd', flush=True)

        with tempfile.TemporaryDirectory(prefix='topk-dispatch-') as temporary:
            path = Path(temporary) / 'events.csv'
            # Test selector dispatch independently of device RNG. LLVM15's
            # r600/Cayman generator emits an overflowing ALU-clause count for
            # the separate random-fill kernel; see docs/LLVM_CAYMAN_RNG_BUG.md.
            x = torch.rand(20, 3, 1344).to('ocl:0')
            with torch.ocl.profile(0, str(path)):
                x.topk(10)
                x.topk(3, dim=1)
            rows = list(csv.DictReader(path.open()))
            names = [r['kernel'] for r in rows]
            assert names.count('topk_parallel') == 1, names
            assert names.count('topk_select') == 1, names
        forbidden = [str(w.message) for w in caught if 'uses CPU' in str(w.message) or 'not currently supported' in str(w.message)]
        assert not forbidden, forbidden
    result = dict(passed=True, cases=checks, autograd=True,
                  autograd_cpu_fallback_warnings=backward_messages,
                  parallel_dispatch_checked=True, fallback_dispatch_checked=True,
                  backend=pytorch_ocl.__file__)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
