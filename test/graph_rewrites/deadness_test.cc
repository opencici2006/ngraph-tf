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

/*******************************************************************************

This test is inspired from the deadness test, mentioned in the commit message of
the deadness analysis found in the below revision

Github repository: https://github.com/tensorflow/tensorflow
Revision: 6619dd5fdcad02f087f5758083e2585bdfef9e78

Quoted from the commit message **
TensorFlow allows nodes to have some live inputs and some dead inputs.  The
executor does not execute these nodes but instead propagates a dead signal to
all their outputs (i.e. these nodes are treated as fully dead).

This is a problem for auto-clustering because it means auto-clustering can kill
nodes that used to be alive.  For instance say before clustering we have a graph
like

digraph {
  Alive0 -> P
  Alive1 -> Q
  Dead -> R
  P -> X
  Q -> X
  Q -> Y
  R -> Y
}

and we cluster P, Q, R, X and Y into a single XLA cluster.

Then after clustering both X and Y are dead because the cluster is a single node
as far as the executor is concerned and said node won't get scheduled if any of
its inputs are dead.

*******************************************************************************/

#include "../test_utilities.h"
#include "gtest/gtest.h"
#include "ngraph_assign_clusters.h"
#include "ngraph_mark_for_clustering.h"
#include "ngraph_utils.h"
#include "tf_graph_writer.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

TEST(DeadnessCheck, livedead1NGRAPH) {
  Scope root = Scope::NewRootScope();

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto C = ops::Placeholder(root, DataType::DT_FLOAT);
  auto pred = ops::Placeholder(root, DataType::DT_BOOL);

  auto S = ops::Switch(root, A, pred);
  auto P = ops::Add(root, A, B);

  auto Q = ops::Add(root, A, C);
  auto R = ops::Sub(root, S.output_true, B);

  auto M = ops::Mul(root, P, Q);
  auto D = ops::RealDiv(root, Q, R);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_NE(
      session.Run(
          {{A, {3.f, 5.f}}, {B, {3.f, 2.f}}, {C, {3.f, 2.f}}, {pred, false}},
          {M, D}, &outputs),
      Status::OK());
}

TEST(DeadnessCheck, livedead1TF) {
  Scope root = Scope::NewRootScope();
  DeactivateNGraph();

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto C = ops::Placeholder(root, DataType::DT_FLOAT);
  auto pred = ops::Placeholder(root, DataType::DT_BOOL);

  auto S = ops::Switch(root, A, pred);
  auto P = ops::Add(root, A, B);

  auto Q = ops::Add(root, A, C);
  auto R = ops::Sub(root, S.output_true, B);

  auto M = ops::Mul(root, P, Q);
  auto D = ops::RealDiv(root, Q, R);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_NE(
      session.Run(
          {{A, {3.f, 5.f}}, {B, {3.f, 2.f}}, {C, {3.f, 2.f}}, {pred, false}},
          {M, D}, &outputs),
      Status::OK());
}

// Graph 1
//
//               A1(#True)[Const]
//              /    \ 
//             /      \
//            /        \ 
//       N1(#P1)[Add]  N2(#P1)[Sub]
// Ops A1, N1 and N2 should be placed in the same cluster
TEST(DeadnessCheck, DTestG1) {
  Scope root = Scope::NewRootScope();

  auto dataX = ops::Placeholder(root.WithOpName("dataX"), DataType::DT_FLOAT);
  auto predX = ops::Placeholder(root.WithOpName("PredX"), DataType::DT_BOOL);
  auto SX = ops::Switch(root.WithOpName("SwitchX"), dataX, predX);

  auto A1 = ops::Const(root.WithOpName("A1"), {3.f, 2.f});
  auto N1_Add = ops::Add(root.WithOpName("N1_Add"), A1, SX.output_false);
  auto N2_Sub = ops::Sub(root.WithOpName("N2_Sub"), A1, SX.output_false);

  //   std::vector<Tensor> outputs;
  //   ClientSession session(root);
  //   ASSERT_EQ(session.Run({{dataX, {3.f, 5.f}}, {predX, false}}, {N1_Add,
  //   N2_Sub},
  //                         &outputs),
  //             Status::OK());

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));
  ASSERT_OK(MarkForClustering(&graph));
  ASSERT_OK(AssignClusters(&graph));

  std::map<std::string, Node*> node_map;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  int N1_Add_cluster, N2_Sub_cluster, A1_cluster;
  ASSERT_OK(GetNodeCluster(node_map["A1"], &A1_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N1_Add"], &N1_Add_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N2_Sub"], &N2_Sub_cluster));

  // A1, N1 and N2 are in same cluster
  ASSERT_EQ(N1_Add_cluster, N2_Sub_cluster);
  ASSERT_EQ(N1_Add_cluster, A1_cluster);
}

// Graph 2
//
//               A1(#True)[Const]
//              /    \      \ 
//             /      \      \ 
//            /        \      \ 
//       N1(#P1)[Add]   \   N2(#P2)[Sub]
//                       \     /
//                        \   /
//                      N3(#P2) [Mul]
// There should be 3 clusters
// Cluster 1 : A1
// Cluster 2 : N2 and N3
// Cluster 3 : N1
TEST(DeadnessCheck, DTestG2) {
  Scope root = Scope::NewRootScope();

  auto dataX = ops::Placeholder(root.WithOpName("dataX"), DataType::DT_FLOAT);
  auto dataY = ops::Placeholder(root.WithOpName("dataY"), DataType::DT_FLOAT);
  auto predX = ops::Placeholder(root.WithOpName("PredX"), DataType::DT_BOOL);
  auto predY = ops::Placeholder(root.WithOpName("PredY"), DataType::DT_BOOL);
  auto SX = ops::Switch(root.WithOpName("SwitchX"), dataX, predX);
  auto SY = ops::Switch(root.WithOpName("SwitchY"), dataY, predY);

  auto A1 = ops::Const(root.WithOpName("A1"), {3.f, 2.f});
  auto N1_Add = ops::Add(root.WithOpName("N1_Add"), SX.output_true, A1);
  auto N2_Sub = ops::Sub(root.WithOpName("N2_Sub"), SY.output_true, A1);
  auto N3_Mul = ops::Mul(root.WithOpName("N3_Mul"), N2_Sub, A1);

  //   std::vector<Tensor> outputs;
  //   ClientSession session(root);
  //   ASSERT_EQ(session.Run({{dataX, {3.f, 5.f}},
  //                          {dataY, {3.f, 2.f}},
  //                          {predX, true},
  //                          {predY, true}},
  //                         {N1_Add, N3_Mul}, &outputs),
  //             Status::OK());

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));
  ASSERT_OK(MarkForClustering(&graph));
  ASSERT_OK(AssignClusters(&graph));

  std::map<std::string, Node*> node_map;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  int N1_Add_cluster, N2_Sub_cluster, A1_cluster, N3_Mul_cluster;
  ASSERT_OK(GetNodeCluster(node_map["A1"], &A1_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N1_Add"], &N1_Add_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N2_Sub"], &N2_Sub_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N3_Mul"], &N3_Mul_cluster));

  // N2 and N3 are in same cluster
  ASSERT_EQ(N2_Sub_cluster, N3_Mul_cluster);
  // A1, N1 and N2 are in different cluster
  ASSERT_NE(N1_Add_cluster, A1_cluster);
  ASSERT_NE(N2_Sub_cluster, A1_cluster);
  ASSERT_NE(N2_Sub_cluster, N1_Add_cluster);
}

// Graph 3
//
// A1(#True)[Pl]   B1(#True)[Const]
//     \          /    \ 
//      \        /      \ 
//       \     /         \ 
//     N1(#True)[Add]  N4(#P1)[Sub]
//            /   \ 
//           /     \ 
//          /       \ 
//  N2(#P1)[Add]   N3(#P1)[Mul]
//
// Ops A1, B2, N1, N2, N3 and N4 should be placed in the same cluster
// P1 is not supported on nGraph so is not clustered
TEST(DeadnessCheck, DTestG3) {
  Scope root = Scope::NewRootScope();

  auto dataX = ops::Placeholder(root.WithOpName("dataX"), DataType::DT_FLOAT);
  auto predX = ops::Placeholder(root.WithOpName("PredX"), DataType::DT_BOOL);
  auto SX = ops::Switch(root.WithOpName("SwitchX"), dataX, predX);

  auto A1 = ops::Const(root.WithOpName("A1"), {3.f, 2.f});
  auto B1 = ops::Const(root.WithOpName("B1"), {3.f, 2.f});
  auto N1_Add = ops::Add(root.WithOpName("N1_Add"), A1, B1);
  auto N2_Add = ops::Add(root.WithOpName("N2_Add"), N1_Add, SX.output_false);
  auto N3_Mul = ops::Mul(root.WithOpName("N3_Mul"), N1_Add, SX.output_false);
  auto N4_Sub = ops::Sub(root.WithOpName("N4_Sub"), A1, SX.output_false);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));
  ASSERT_OK(MarkForClustering(&graph));
  ASSERT_OK(AssignClusters(&graph));

  std::map<std::string, Node*> node_map;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  int N1_Add_cluster, N2_Add_cluster, A1_cluster, N3_Mul_cluster,
      N4_Sub_cluster, B1_cluster;
  ASSERT_OK(GetNodeCluster(node_map["A1"], &A1_cluster));
  ASSERT_OK(GetNodeCluster(node_map["B1"], &B1_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N1_Add"], &N1_Add_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N2_Add"], &N2_Add_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N3_Mul"], &N3_Mul_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N4_Sub"], &N4_Sub_cluster));

  // A1, B1, N1, N2, N3 and N4 are in the same cluster
  ASSERT_EQ(A1_cluster, B1_cluster);
  ASSERT_EQ(N1_Add_cluster, N2_Add_cluster);
  ASSERT_EQ(N3_Mul_cluster, N4_Sub_cluster);
  ASSERT_EQ(A1_cluster, N1_Add_cluster);
  ASSERT_EQ(N1_Add_cluster, N3_Mul_cluster);
}

// Graph 4
// Ops A1, N1 and N2 should be placed in the same cluster
//
// A1(#True)[Const]   B1(#True)[Const]
//     \          /    \ 
//      \        /      \ 
//       \     /         \ 
//     N1(#True)[Add]  N4(#P2)[Sub]
//            /   \ 
//           /     \ 
//          /       \ 
//  N2(#P1)[Add]   N3(#P1)[Mul]
//
// P1 is not supported on nGraph so is not clustered
// There will be 3 clusters
// Cluster 1: B1
// Cluster 2: A1, N1, N2, N3
// Cluster 3: N4
TEST(DeadnessCheck, DTestG4) {
  Scope root = Scope::NewRootScope();

  auto dataX = ops::Placeholder(root.WithOpName("dataX"), DataType::DT_FLOAT);
  auto predX = ops::Placeholder(root.WithOpName("PredX"), DataType::DT_BOOL);
  auto SX = ops::Switch(root.WithOpName("SwitchX"), dataX, predX);
  auto dataY = ops::Placeholder(root.WithOpName("dataY"), DataType::DT_FLOAT);
  auto predY = ops::Placeholder(root.WithOpName("PredY"), DataType::DT_BOOL);
  auto SY = ops::Switch(root.WithOpName("SwitchY"), dataY, predY);

  auto A1 = ops::Const(root.WithOpName("A1"), {3.f, 2.f});
  auto B1 = ops::Const(root.WithOpName("B1"), {3.f, 2.f});
  auto N1_Add = ops::Add(root.WithOpName("N1_Add"), A1, B1);
  auto N2_Add = ops::Add(root.WithOpName("N2_Add"), N1_Add, SX.output_false);
  auto N3_Mul = ops::Mul(root.WithOpName("N3_Mul"), N1_Add, SX.output_false);
  auto N4_Sub = ops::Sub(root.WithOpName("N4_Sub"), B1, SY.output_false);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));
  ASSERT_OK(MarkForClustering(&graph));
  ASSERT_OK(AssignClusters(&graph));

  std::map<std::string, Node*> node_map;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  int N1_Add_cluster, N2_Add_cluster, A1_cluster, N3_Mul_cluster,
      N4_Sub_cluster, B1_cluster;
  ASSERT_OK(GetNodeCluster(node_map["A1"], &A1_cluster));
  ASSERT_OK(GetNodeCluster(node_map["B1"], &B1_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N1_Add"], &N1_Add_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N2_Add"], &N2_Add_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N3_Mul"], &N3_Mul_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N4_Sub"], &N4_Sub_cluster));

  // A1, N1, N2, N3 are in the same cluster
  ASSERT_EQ(A1_cluster, N1_Add_cluster);
  ASSERT_EQ(N2_Add_cluster, N3_Mul_cluster);
  ASSERT_EQ(A1_cluster, N2_Add_cluster);

  // A1, B1 and N4 are in different clusters
  ASSERT_NE(A1_cluster, B1_cluster);
  ASSERT_NE(A1_cluster, N4_Sub_cluster);
  ASSERT_NE(B1_cluster, N4_Sub_cluster);

  //   std::vector<Tensor> outputs;
  //   ClientSession session(root);
  //   ASSERT_EQ(session.Run({{dataX, {3.f, 5.f}},
  //                          {dataY, {3.f, 5.f}},
  //                          {predX, false},
  //                          {predY, false},
  //                          {P1, {3.f, 5.f}}},
  //                         {N2_Add, N3_Mul, N4_Sub}, &outputs),
  //             Status::OK());
}

// Graph 5
//            SX(#True)[Switch]      SY(#True)[Switch]
//                   \                   /  \ 
//                    \(X)          (~Y)/    \(Y)
//  A(#True)[Const]    \-> N1(X & ~Y)[Add]   N5(Y)[Add]<----- B(#True)[Pl]
//         \                   |                |              |
//          \                  |                |              |
//           \----------->N2(X & ~Y)[Mul]    N6(Y)[Mul]<-------|
//            \                |
//             \               |
// SZ(#True)----\-------->N3(Z & X & ~Y)[Sub]
//  [Switch]     \             |
//                \            |
//                 \-->N4(Z & X & ~Y)[Mul]
//
// There should be 4 clusters
// Cluster 1 : A1
// Cluster 2 : N1 and N2
// Cluster 3 : N3 and N4
// Cluster 4 : B, N5 and N6
TEST(DeadnessCheck, DTestPl) {
  Scope root = Scope::NewRootScope();

  auto dataX = ops::Placeholder(root.WithOpName("dataX"), DataType::DT_FLOAT);
  auto dataY = ops::Placeholder(root.WithOpName("dataY"), DataType::DT_FLOAT);
  auto dataZ = ops::Placeholder(root.WithOpName("dataZ"), DataType::DT_FLOAT);
  auto predX = ops::Placeholder(root.WithOpName("PredX"), DataType::DT_BOOL);
  auto predY = ops::Placeholder(root.WithOpName("PredY"), DataType::DT_BOOL);
  auto predZ = ops::Placeholder(root.WithOpName("PredZ"), DataType::DT_BOOL);

  auto SX = ops::Switch(root.WithOpName("SwitchX"), dataX, predX);
  auto SY = ops::Switch(root.WithOpName("SwitchY"), dataY, predY);
  auto SZ = ops::Switch(root.WithOpName("SwitchZ"), dataZ, predZ);

  auto A = ops::Const(root.WithOpName("A"), {3.f, 2.f});
  auto B = ops::Const(root.WithOpName("B"), {3.f, 2.f});
  auto N1_Add =
      ops::Add(root.WithOpName("N1_Add"), SX.output_true, SY.output_false);
  auto N2_Mul = ops::Mul(root.WithOpName("N2_Mul"), N1_Add, A);
  auto N3_Sub = ops::Sub(root.WithOpName("N3_Sub"), SZ.output_true, N2_Mul);
  auto N4_Mul = ops::Mul(root.WithOpName("N4_Mul"), N3_Sub, A);
  auto N5_Add = ops::Add(root.WithOpName("N5_Add"), SY.output_true, B);
  auto N6_Mul = ops::Mul(root.WithOpName("N6_Mul"), N5_Add, B);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));
  ASSERT_OK(MarkForClustering(&graph));
  ASSERT_OK(AssignClusters(&graph));

  std::map<std::string, Node*> node_map;

  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  int N1_Add_cluster, N2_Mul_cluster, N3_Sub_cluster, N4_Mul_cluster,
      N5_Add_cluster, N6_Mul_cluster, A_cluster, B_cluster;

  ASSERT_OK(GetNodeCluster(node_map["A"], &A_cluster));
  ASSERT_OK(GetNodeCluster(node_map["B"], &B_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N1_Add"], &N1_Add_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N2_Mul"], &N2_Mul_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N3_Sub"], &N3_Sub_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N4_Mul"], &N4_Mul_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N5_Add"], &N5_Add_cluster));
  ASSERT_OK(GetNodeCluster(node_map["N6_Mul"], &N6_Mul_cluster));

  // N1 and N2 are in the same cluster
  ASSERT_EQ(N1_Add_cluster, N2_Mul_cluster);
  // N3 and N4 are in same cluster
  ASSERT_EQ(N3_Sub_cluster, N4_Mul_cluster);
  // B, N5 and N6 are in same cluster
  ASSERT_EQ(N5_Add_cluster, N6_Mul_cluster);
  ASSERT_EQ(N5_Add_cluster, B_cluster);

  // N1 and N3 are in different cluster
  ASSERT_NE(N1_Add_cluster, N3_Sub_cluster);
  // N1 and N5 are in different cluster
  ASSERT_NE(N1_Add_cluster, N5_Add_cluster);
  // N3 and N5 are in differenct cluster
  ASSERT_NE(N3_Sub_cluster, N5_Add_cluster);
  // A, N1, N3 and N5 are a different cluster
  ASSERT_NE(A_cluster, N1_Add_cluster);
  ASSERT_NE(A_cluster, N3_Sub_cluster);
  ASSERT_NE(A_cluster, N5_Add_cluster);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
