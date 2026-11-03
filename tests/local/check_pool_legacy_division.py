"""CPU parity for the pool-with-indices path used by YOLO inference mode."""
import warnings
import torch
import pytorch_ocl

torch.set_num_threads(2)
torch.manual_seed(6850)
device = 'ocl:0'
checks = 0


def compare(x, kernel, stride, padding, dilation=1, ceil=False):
    global checks
    expected = torch.nn.functional.max_pool2d(x, kernel, stride, padding, dilation, ceil, return_indices=True)
    for mode in (torch.no_grad, torch.inference_mode):
        with mode():
            actual = torch.nn.functional.max_pool2d(x.to(device), kernel, stride, padding, dilation, ceil,
                                                  return_indices=True)
            torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=0, atol=0)
        checks += 1


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    for shape in [(1, 2, 8, 8), (1, 2, 5, 9), (4, 128, 5, 9), (2, 3, 9, 11),
                  (3, 9, 11), (0, 3, 9, 11)]:
        x = torch.randn(shape)
        for params in [(5, 1, 2), (3, 2, 1, 1, True), (3, 1, 1, 2, False), (2, 2, 0)]:
            compare(x, *params)
    compare(torch.randn(2, 3, 11, 9).transpose(-1, -2), 5, 1, 2)
    compare(torch.randn(3, 3, 9, 11)[1:], 3, 2, 1)
    special = torch.tensor([[[[float('nan'), -float('inf'), 2.],
                              [float('nan'), -float('inf'), 2.],
                              [0., 0., 0.]]]])
    compare(special, 2, 1, 0)
    compare(torch.full((1, 2, 5, 9), -float('inf')), 5, 1, 2)
    compare(torch.zeros((1, 2, 5, 9)), 5, 1, 2)
    x = torch.randn(2, 3, 9, 11)
    expected = torch.nn.functional.max_pool2d(x, 5, 1, 2, return_indices=True)
    shape = expected[0].shape
    values_base = torch.full((*shape, 2), -12345., device=device)
    indices_base = torch.full((*shape, 2), -12345, dtype=torch.long, device=device)
    values, indices = values_base[..., 1], indices_base[..., 1]
    strides = values.stride(), indices.stride()
    with torch.inference_mode():
        torch.ops.aten.max_pool2d_with_indices.out(x.to(device), [5, 5], [1, 1], [2, 2], [1, 1], False,
                                                 out=values, indices=indices)
    assert (values.stride(), indices.stride()) == strides
    torch.testing.assert_close(values.cpu(), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(indices.cpu(), expected[1], rtol=0, atol=0)
    assert (values_base.cpu()[..., 0] == -12345).all()
    assert (indices_base.cpu()[..., 0] == -12345).all()
    checks += 1
    bad = [str(w.message) for w in caught if 'uses CPU' in str(w.message) or 'not currently supported' in str(w.message)]
    assert not bad, bad
print(f'PASS {checks} pooling CPU comparisons, including rectangular inference and guarded output views', flush=True)
