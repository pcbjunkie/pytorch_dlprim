"""Check hardtanh/ReLU6 boundary derivatives and expanded upstream gradients."""
import torch
import pytorch_ocl

torch.set_num_threads(2)
for lower, upper in ((0., 6.), (-1., 1.)):
    for inplace in (False, True):
        for reduced in (False, True):
            results = []
            for device in ('cpu', 'ocl:0'):
                x = torch.tensor([lower-1, lower, (lower+upper)/2, upper, upper+1],
                                 device=device, requires_grad=True)
                y = torch.nn.functional.hardtanh(x.clone() if inplace else x,
                                                lower, upper, inplace=inplace)
                if reduced:
                    y.sum().backward()
                else:
                    y.backward(torch.ones_like(y))
                results.append((y.detach().cpu(), x.grad.detach().cpu()))
            for expected, actual in zip(results[0], results[1]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
print('PASS: hardtanh/ReLU6 boundaries, inplace forms and expanded gradients.')
