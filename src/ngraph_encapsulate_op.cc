/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tf = tensorflow;

namespace ngraph_bridge {

REGISTER_OP("NGraphEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    //.Attr("function: func")
    .Attr("ngraph_cluster: int")
    .SetIsStateful()
    .Doc("nGraph Encapsulation Op. For use by the nGraph JIT only.");

class NGraphEncapsulateOp : public tf::OpKernel {
 public:
  explicit NGraphEncapsulateOp(tf::OpKernelConstruction* ctx)
      : tf::OpKernel(ctx) {
    // DataTypeVector constant_types;
    // OP_REQUIRES_OK(ctx, ctx->GetAttr("Tconstants", &constant_types));
    // num_constant_args_ = constant_types.size();
    // OP_REQUIRES_OK(ctx, ctx->GetAttr("Nresources", &num_resource_args_));
    VLOG(0) << "NGraphEncapsulateOp::Number of inputs: " << ctx->num_inputs();
    VLOG(0) << "NGraphEncapsulateOp::Number of outputs: " << ctx->num_outputs();

    // Get the functions
    auto function_lib = ctx->function_library();
    auto function_lib_def = function_lib->GetFunctionLibraryDefinition();
    VLOG(0) << "Number of functions: " << function_lib_def->num_functions();
  }
  ~NGraphEncapsulateOp() override {
    // d-tor
  }
  void Compute(tf::OpKernelContext* ctx) override {
    VLOG(0) << "NGraphMulOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    VLOG(0) << "Inputs: " << ctx->num_inputs()
            << " Outputs: " << ctx->num_outputs();
    // Get the inputs
    const tf::Tensor& input_tensor_1 = ctx->input(0);
    const tf::Tensor& input_tensor_2 = ctx->input(1);

    // DO the Math

    // Save the output
    // Create an output tensor
    tf::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor_1.shape(), &output_tensor));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, input_tensor_1.shape(), &output_tensor));
  }
};

}  // namespace ngraph_bridge

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("NGraphEncapsulate").Device("NGRAPH_CPU"),
                        ngraph_bridge::NGraphEncapsulateOp);
}
