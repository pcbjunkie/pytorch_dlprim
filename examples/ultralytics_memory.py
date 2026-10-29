"""Optional Ultralytics 8.3.0 progress adapter; no GPU synchronization or polling.

Place this mixin before the detection/pose trainer in its method resolution order.
The progress field is the process allocator's high-water mark since startup or
the last explicit reset, in decimal GB. It is neither whole-card VRAM nor a
per-batch peak. Driver counters are unavailable through the OpenCL allocator.
"""
import json
import os
from pathlib import Path

import torch


def allocator_snapshot(device=None):
    stats = torch.ocl.memory_stats(device)
    reserve = os.environ.get('R600_POOL_RESERVE_MB', '')
    configured = int(reserve) * 1024**2 if reserve.isdecimal() and 1 <= int(reserve) <= 1024 else None
    return {
        'units': 'bytes',
        'peak_scope': 'since_process_start_or_explicit_reset',
        'live_tensor_bytes': stats['allocated_bytes.all.current'],
        'backend_reserved_bytes': stats['reserved_bytes.all.current'],
        'peak_live_tensor_bytes': stats['allocated_bytes.all.peak'],
        'peak_backend_reserved_bytes': stats['reserved_bytes.all.peak'],
        'cached_bytes': stats['cached_bytes.all.current'],
        'deferred_bytes': stats.get('deferred_bytes.all.current', 0),
        'peak_deferred_bytes': stats.get('deferred_bytes.all.peak', 0),
        # A request may fall back to a smaller pool. Never present it as measured.
        'requested_driver_pool_reserve_bytes': configured,
        'driver_pool_capacity_bytes': None,
        'driver_pool_occupancy_bytes': None,
        'whole_card_vram_used_bytes': None,
    }


class OpenCLMemoryMixin:
    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        self.add_callback('on_train_epoch_end', self._record_opencl_memory)
        from ultralytics.utils import LOGGER
        LOGGER.info('OCLpeak_GB: peak backend-reserved storage since process start/reset, '
                    'decimal GB; excludes driver pool, other allocations and other processes.')

    def _get_memory(self):
        return torch.ocl.max_memory_reserved(self.device) / 1e9

    def progress_string(self):
        return super().progress_string().replace('GPU_mem', 'OCLpeak_GB')

    @staticmethod
    def _record_opencl_memory(trainer):
        row = allocator_snapshot(trainer.device)
        row['epoch'] = trainer.epoch + 1
        with (Path(trainer.save_dir) / 'opencl-memory.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
