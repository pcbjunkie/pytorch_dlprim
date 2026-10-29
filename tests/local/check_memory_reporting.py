"""Real allocator/view/retirement peaks and the optional YOLO progress adapter."""
import gc
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytorch_ocl

spec = importlib.util.spec_from_file_location('ultralytics_memory',
    Path(__file__).resolve().parents[2] / 'examples/ultralytics_memory.py')
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)
torch.set_num_threads(2)
torch.ocl.empty_cache()
torch.ocl.synchronize()
torch.ocl.reset_peak_memory_stats()
base = reporting.allocator_snapshot(0)
x = torch.ones(3 * 1024 * 1024 // 4, device='ocl:0')
torch.ocl.synchronize()
alias = x[1::2]
allocated = reporting.allocator_snapshot(0)
assert allocated['live_tensor_bytes'] - base['live_tensor_bytes'] == x.numel() * 4
del x
gc.collect()
assert reporting.allocator_snapshot(0)['live_tensor_bytes'] == allocated['live_tensor_bytes']
torch.testing.assert_close(alias[:7].cpu(), torch.ones(7))
del alias
gc.collect()
torch.ocl.synchronize()
torch.ocl.empty_cache()
released = reporting.allocator_snapshot(0)
assert released['live_tensor_bytes'] == base['live_tensor_bytes']
assert released['backend_reserved_bytes'] == base['backend_reserved_bytes']
assert released['peak_backend_reserved_bytes'] >= allocated['backend_reserved_bytes']

class Parent:
    def progress_string(self):
        return 'Epoch GPU_mem loss'

class Trainer(reporting.OpenCLMemoryMixin, Parent):
    device = torch.device('ocl:0')

trainer = Trainer()
# Any added queue synchronization or driver polling is an error in progress reads.
with patch.object(torch.ocl, 'synchronize', side_effect=AssertionError('blocking telemetry')):
    assert trainer._get_memory() == released['peak_backend_reserved_bytes'] / 1e9
    assert 'OCLpeak_GB' in trainer.progress_string()
    assert 'GPU_mem' not in trainer.progress_string()
    with tempfile.TemporaryDirectory() as temp:
        trainer.save_dir = temp
        trainer.epoch = 2
        trainer._record_opencl_memory(trainer)
        row = json.loads((Path(temp) / 'opencl-memory.jsonl').read_text())
        assert row['epoch'] == 3 and row['units'] == 'bytes'
        assert row['driver_pool_capacity_bytes'] is None
        assert row['whole_card_vram_used_bytes'] is None
torch.ocl.reset_peak_memory_stats()
reset = reporting.allocator_snapshot(0)
assert reset['peak_live_tensor_bytes'] == reset['live_tensor_bytes']
assert reset['peak_backend_reserved_bytes'] == reset['backend_reserved_bytes']
print('PASS live/alias/release/peak/reset reporting, decimal GB, nonblocking progress, unknown driver counters')
