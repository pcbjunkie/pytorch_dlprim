#include "utils.h"
#include "batch_pointwise.h"
#include <dlprim/core/pointwise.hpp>

namespace ptdlprim {
// Preserve separate FP32 multiplies before addition rather than deliberately
// introducing an FMA with a different rounding rule. One launch per tensor.
void ema_update_(at::TensorList averages,at::TensorList values,double decay)
{
    GUARD;
    TORCH_CHECK(averages.size()==values.size(),"EMA tensor list lengths differ");
    TORCH_CHECK(std::isfinite(decay) && decay>=0 && decay<=1,"EMA decay must be finite and in [0,1]");
    for(size_t i=0;i<averages.size();i++) {
        auto &a=averages[i];auto &v=values[i];
        TORCH_CHECK(!at::GradMode::is_enabled() || (!a.requires_grad() && !v.requires_grad()),"EMA update is not differentiable; use no_grad");
        TORCH_CHECK(a.device().type()==OpenCLDeviceType && a.device()==v.device(),"EMA tensors must share an OpenCL device");
        TORCH_CHECK(a.scalar_type()==at::kFloat && v.scalar_type()==at::kFloat,"Fused EMA requires FP32");
        TORCH_CHECK(a.sizes()==v.sizes(),"EMA shapes differ");
        TORCH_CHECK(a.is_contiguous() && v.is_contiguous(),"Fused EMA requires contiguous tensors");
        TORCH_CHECK(!a.is_alias_of(v),"EMA average and source must not alias");
    }
    if(batch_pointwise(averages,values,averages,decay,1.0-decay,c10::nullopt,BatchExpression::ema,true))return;
    for(size_t i=0;i<averages.size();i++) {
        auto &a=averages[i];auto &v=values[i];
        if(!a.numel()){a.unsafeGetTensorImpl()->bump_version();continue;}
        auto q=getExecutionContext(a);
        dlprim::core::pointwise_operation({todp(a),todp(v)},{todp(a)},
             {decay,1.0-decay},
             "volatile float previous = x0*w0; volatile float current = x1*w1; y0 = previous + current;",q);
        a.unsafeGetTensorImpl()->bump_version();
        sync_if_needed(a.device());
    }
}
}
TORCH_LIBRARY_FRAGMENT(dlprim,m) {
    m.def("ema_update_(Tensor(a!)[] averages, Tensor[] values, float decay) -> ()");
}
TORCH_LIBRARY_IMPL(dlprim,PrivateUse1,m) {
    m.impl("ema_update_",&ptdlprim::ema_update_);
}

TORCH_LIBRARY_IMPL(dlprim,AutogradPrivateUse1,m) {
    m.impl("ema_update_",torch::CppFunction::makeFallthrough());
}
