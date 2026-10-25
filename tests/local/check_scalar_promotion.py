"""Regression checks for integer counts combined with floating scalar add/sub."""
import torch
import pytorch_ocl

torch.backends.mkldnn.enabled = False
torch.set_num_threads(2)
for dtype in (torch.int32, torch.int64, torch.float32):
    cpu = torch.tensor([[0, 1, 17], [-2, 4, 8]], dtype=dtype).t()
    gpu = cpu.to('ocl:0')
    for scalar in (1e-9, 0.25, -2.5):
        for name, operation in (
            ('add', lambda x: x + scalar),
            ('reverse_add', lambda x: scalar + x),
            ('subtract', lambda x: x - scalar),
            ('reverse_subtract', lambda x: scalar - x),
        ):
            expected = operation(cpu)
            actual = operation(gpu).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-12,
                                       msg=lambda msg: f'{dtype} {name} {scalar}: {msg}')
print('PASS: scalar promotion, signs, zero counts and noncontiguous inputs.')
