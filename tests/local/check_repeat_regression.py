"""Device repeat must preserve values and gradients without CPU view fallback."""
import warnings
import torch,pytorch_ocl

torch.set_num_threads(2)
with warnings.catch_warnings(record=True) as caught:
 warnings.simplefilter('always')
 for shape,repeats in [((2,84,1),(1,1,5)),((2,3),(2,3)),((2,1,3),(1,4,2)),((3,),(2,3)),((2,3),(0,2)),((),(2,3))]:
  x=torch.arange(torch.tensor(shape).prod().item() if shape else 1,dtype=torch.float32).reshape(shape).requires_grad_()
  g=x.detach().to('ocl:0').requires_grad_()
  expected=x.repeat(*repeats); actual=g.repeat(*repeats)
  torch.testing.assert_close(actual.cpu(),expected)
  expected.sum().backward();actual.sum().backward()
  torch.testing.assert_close(g.grad.cpu(),x.grad)
  print(shape,repeats,'values and gradients PASS',flush=True)
 x=torch.randn(3,5).t()
 torch.testing.assert_close(x.to('ocl:0').repeat(2,3).cpu(),x.repeat(2,3))
 assert not [str(w.message) for w in caught if 'falling back' in str(w.message) or 'not currently supported' in str(w.message)]
print('PASS: repeat values/gradients, empty/scalar/noncontiguous inputs, no fallback')
