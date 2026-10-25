import torch
from types import SimpleNamespace
from .pt_ocl import *

def _device_index(device):
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type not in ('ocl', 'privateuseone'):
            raise ValueError('Expected an OpenCL device, got ' + str(device))
        if device.index is None:
            return impl_current_device()
        return device.index
    return -1


class _OCL:
    class profile:
        """ Enables profiling for a ocl device and saves result log"""
        def __init__(self,device,path = None):
            """ if path is not None profiling is enabled and result is saved in csv format to path"""
            self._device_id = _device_index(device)
            self._path = path

        def __enter__(self):
            if self._path is not None:
                impl_start_profiling(self._device_id)
        
        def __exit__(self, type, value, traceback):
            if self._path is not None:
                impl_stop_profiling(self._device_id,self._path)

    def enable_profiling(device):
        impl_enable_profiling(_device_index(device))
        
    class device:
        def __init__(self, device):
            self.idx = _device_index(device)
            if self.idx < 0:
                self.idx = impl_current_device()

        def __enter__(self):
            self.prev_idx = impl_current_device()
            impl_set_device(self.idx)
            return self
        
        def __exit__(self, type, value, traceback):
            impl_set_device(self.prev_idx)
            return False

    @staticmethod
    def synchronize(dev = None):
        if dev is None:
            impl_synchronize_device(-1)
        else:
            impl_synchronize_device(_device_index(dev))

    @staticmethod
    def manual_seed_all(seed:int):
        impl_seed_all(seed)

    @staticmethod
    def _is_in_bad_fork():
        return impl_is_bad_fork()

    @staticmethod
    def empty_cache():
        impl_empty_cache()

    @staticmethod
    def memory_stats(device=None):
        """Allocator counters only; excludes driver allocations and other processes.

        Allocated counts requested live storage bytes (views are not counted twice).
        Reserved counts rounded live and cached OpenCL buffers.
        """
        index = impl_current_device() if device is None else _device_index(device)
        return impl_memory_stats(index, False)

    @staticmethod
    def memory_allocated(device=None):
        return _OCL.memory_stats(device)['allocated_bytes.all.current']

    @staticmethod
    def memory_reserved(device=None):
        return _OCL.memory_stats(device)['reserved_bytes.all.current']

    @staticmethod
    def max_memory_allocated(device=None):
        return _OCL.memory_stats(device)['allocated_bytes.all.peak']

    @staticmethod
    def max_memory_reserved(device=None):
        return _OCL.memory_stats(device)['reserved_bytes.all.peak']

    @staticmethod
    def reset_peak_memory_stats(device=None):
        index = impl_current_device() if device is None else _device_index(device)
        impl_memory_stats(index, True)

    @staticmethod
    def device_count():
        return impl_device_count()

    @staticmethod
    def current_device():
        return impl_current_device()

    @staticmethod
    def set_device(device):
        impl_set_device(_device_index(device))

    @staticmethod
    def get_device_properties(device=None):
        index = impl_current_device() if device is None else _device_index(device)
        return SimpleNamespace(**impl_device_properties(index))

    @staticmethod
    def get_device_name(device=None):
        return _OCL.get_device_properties(device).name

    @staticmethod
    def is_available():
        try:
            return impl_device_count() > 0
        except RuntimeError:
            return False

def synchronize(dev=None):
    _OCL.synchronize(dev)

def manual_seed_all(seed):
    _OCL.manual_seed_all(seed)

def empty_cache():
    _OCL.empty_cache()


try: 
    torch.utils.rename_privateuse1_backend('ocl')
    torch._register_device_module("ocl", _OCL)
except:
    pass
