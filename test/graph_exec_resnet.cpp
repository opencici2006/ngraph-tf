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
#include "gtest/gtest.h"

#include "ngraph_builder.h"
#include "ngraph_utils.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

TEST(graph_exec, resnet) {
  GraphDef gdef;
  // auto status = ReadTextProto(Env::Default(), "test_py.pbtxt",
  // &gdef);
  auto status = ReadBinaryProto(Env::Default(),
                                "../../examples/tmp/frozen_model.pb", &gdef);
  // ReadTextProto(Env::Default(), "test_launch_op.pbtxt", &gdef);
  ASSERT_TRUE(status == Status::OK()) << "Can't read protobuf graph";

  Graph input_graph(OpRegistry::Global());

  GraphConstructorOptions opts;
  // Set the allow_internal_ops to true so that graphs with node names such as
  // _arg_Placeholder_1_0_1_0_arg are allowed. These op names are generated
  // during the graph rewrite passes and considered internal
  opts.allow_internal_ops = true;

  ASSERT_EQ(ConvertGraphDefToGraph(opts, gdef, &input_graph), Status::OK());
  // Create the inputs for this graph

  int bs = 128;
  int ht = 224;
  int wd = 224;
  int ch = 3;

  Tensor x(DT_FLOAT, TensorShape({bs, ht, wd, ch}));
  std::vector<TensorShape> inputs = {x.shape()};
  std::vector<const Tensor*> static_input_map(1, nullptr);

  shared_ptr<ng::Function> ng_function;
  ASSERT_EQ(Status::OK(),
            ngraph_bridge::Builder::TranslateGraph(inputs, static_input_map,
                                                   &input_graph, ng_function));

  // Create the nGraph backend
  auto backend = ng::runtime::Backend::create("CPU");

  // Allocate tensors for arguments a, b, c
  ng::Shape ng_shape_x(x.shape().dims());
  for (int i = 0; i < x.shape().dims(); ++i) {
    ng_shape_x[i] = x.shape().dim_size(i);
  }

  auto t_x = backend->create_tensor(ng::element::f32, ng_shape_x);
  float* v_x = (float*)calloc(bs * ht * wd * ch, sizeof(float));
  t_x->write(&v_x, 0, bs * ht * wd * ch);

  // Allocate tensor for the result(s)
  vector<shared_ptr<ng::runtime::TensorView>> outputs;
  for (auto i = 0; i < ng_function->get_output_size(); i++) {
    auto shape = ng_function->get_output_shape(i);
    auto elem_type = ng_function->get_output_element_type(i);
    auto t_result = backend->create_tensor(elem_type, shape);
    outputs.push_back(t_result);
  }

  // Execute the nGraph function.
  cout << "Calling nGraph function\n";
  backend->call(ng_function, outputs, {t_x});

  for (auto i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor(cout, ng_function->get_output_op(i)->get_name(), outputs[i]);
    cout << endl;
  }
  // Add the validation logic
  // TODO

  free(v_x);
}

}  // namespace ngraph_bridge

}  // namespace tensorflwo
