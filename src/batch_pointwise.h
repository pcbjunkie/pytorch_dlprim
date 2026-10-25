#pragma once
#include "utils.h"
namespace ptdlprim {
enum class BatchExpression {ema,add,mul_scalar,mul_tensor};
// Returns false without enqueuing or mutating anything if the fast path does
// not preserve layout/device/alias/autograd semantics for these inputs.
bool batch_pointwise(at::TensorList a,at::TensorList b,at::TensorList outputs,
                     double w0,double w1,c10::optional<at::Tensor> scalar,
                     BatchExpression expression,bool bump_versions);
}
