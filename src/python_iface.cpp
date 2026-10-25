//#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include "CLTensor.h"
#include "utils.h"
#include <dlprim/core/util.hpp>
#include <c10/core/impl/VirtualGuardImpl.h>
namespace ptdlprim {


    using namespace torch;
    using torch::autograd::tensor_list;
    using torch::autograd::AutogradContext;


    using c10::Device;
    using c10::DeviceType;
    
    Device get_custom_device(int id) 
    {
      return Device(OpenCLDeviceType, id);
    }
    unsigned device_count()
    {
        return CLContextManager::count();
    }
    int current_device()
    {
        c10::impl::VirtualGuardImpl guard(OpenCLDeviceType);
        return guard.getDevice().index();
    }
    void set_device(int id)
    {
        TORCH_CHECK(id >= 0 && id < int(device_count()), "Invalid OpenCL device index: ", id);
        c10::impl::VirtualGuardImpl guard(OpenCLDeviceType);
        guard.setDevice(Device(OpenCLDeviceType, id));
    }
    pybind11::dict device_properties(int id)
    {
        TORCH_CHECK(id >= 0 && id < int(device_count()), "Invalid OpenCL device index: ", id);
        auto ctx = CLContextManager::getContext(id);
        auto device = ctx.device();
        pybind11::dict properties;
        properties["name"] = device.getInfo<CL_DEVICE_NAME>();
        properties["total_memory"] = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
        properties["max_memory_allocation"] = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        properties["multi_processor_count"] = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        properties["platform"] = ctx.platform().getInfo<CL_PLATFORM_NAME>();
        properties["vendor"] = device.getInfo<CL_DEVICE_VENDOR>();
        properties["driver_version"] = device.getInfo<CL_DRIVER_VERSION>();
        properties["is_gpu"] = bool(device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU);
        return properties;
    }
    void seed_all(uint64_t seed)
    {
        int N = CLContextManager::instance().count();
        for(int i=0;i<N;i++) {
            CLContextManager::rng_state(i).seed(seed);
            if(CLContextManager::is_ready(i)) {
                CLContextManager::getCommandQueue(i).finish();
            }
        }

    }
    void synchronize_device(int index)
    {
        if(index == -1) {
            int N = CLContextManager::instance().count();
            for(int i=0;i<N;i++) {
                if(CLContextManager::is_ready(i)) {
                    CLContextManager::getCommandQueue(i).finish();
                }
            }
        }
        else {
            CLContextManager::getCommandQueue(index).finish();
        }
    }
    bool is_bad_fork()
    {
        return CLContextManager::bad_fork();
    }
    void empty_cache()
    {
        int N = CLContextManager::instance().count();
        for(int i=0;i<N;i++) {
            CLContextManager::clear(i);
        }
    }

    bool enable_profiling(int device)
    {
        return CLContextManager::enable_profiling(device);
    }
    void start_profiling(int device)
    {
        CLContextManager::start_profiling(device);
    }
    void stop_profiling(int device,std::string log)
    {
        CLContextManager::stop_profiling(device,log);
    }
}


// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(pt_ocl, m) {
    m.def("impl_memory_stats", &ptdlprim::CLContextManager::memory_stats, "Allocator byte counters; reset=True resets peaks to current values");
    m.def("impl_device_count", &ptdlprim::device_count, "Number of visible OpenCL devices");
    m.def("impl_current_device", &ptdlprim::current_device, "Current OpenCL device index");
    m.def("impl_set_device", &ptdlprim::set_device, "Set the current OpenCL device index");
    m.def("impl_device_properties", &ptdlprim::device_properties, "OpenCL device properties");
    m.def("impl_custom_device", &ptdlprim::get_custom_device, "get custom device object");
    m.def("impl_seed_all", &ptdlprim::seed_all, "Seed all devices");
    m.def("impl_synchronize_device",&ptdlprim::synchronize_device,"Sychronize device");
    m.def("impl_is_bad_fork",&ptdlprim::is_bad_fork,"True of forked process");
    m.def("impl_empty_cache",&ptdlprim::empty_cache,"Clear all device cache");
    m.def("impl_enable_profiling",&ptdlprim::enable_profiling,"Internal function use torch.ocl.enable_profiling(device)");
    m.def("impl_start_profiling",&ptdlprim::start_profiling,"Internal function use torch.ocl.profile");
    m.def("impl_stop_profiling",&ptdlprim::stop_profiling,"Internal function use torch.ocl.profile");
}
