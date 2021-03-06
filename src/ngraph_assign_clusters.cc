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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/device_name_utils.h"

#include "ngraph_assign_clusters.h"
#include "ngraph_cluster_manager.h"
#include "ngraph_log.h"
#include "ngraph_mark_for_clustering.h"
#include "ngraph_utils.h"
#include "tf_graphcycles.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// The clustering pass performs a greedy search for groups of nGraph-marked ops
// that can be coalesced into a single nGraph graph, and assigns each such
// group a unique identifier called a "cluster ID".
//
// For example, consider the following graph:
//
//       N1
//      /  \
//    N2    N5
//    / \    |
//   N3  N4 N6
//           |
//          N7
//
// If nodes N1, N2, N3, N4, N6, and N7 are all marked, but N6 is unmarked, the
// clustering pass will assign nodes N1, N2, N3, and N4 to one cluster, and
// nodes N6 and N7 to another.
//
// After clustering, it must be the case that:
//
//   (1) every marked node is assigned to exactly one cluster;
//   (2) no unmarked node is assigned to any cluster;
//   (3) for every pair (N1,N2) of nodes where N1 and N2 are in the same
//       cluster, there is no path from N1 to N2 traversing any node N3 that
//       is _not_ in the same cluster as N1 and N2 (in other words,
//       data/control flow cannot "re-enter" the cluster).
//
// Given the above constraints, we try to find the "biggest" clusters we can.
//
// The assigned cluster index is represented by the "_ngraph_cluster"
// attribute, which has integer type.
//
// Assumption: the "MarkForClustering" pass (ngraph_mark_for_clustering.cc) has
// already been run. This attaches the "_ngraph_marked_for_clustering"
// attribute to ops which we will cluster.
//
// TODO(amprocte): Say more about the algorithm.
//

namespace {
struct Cluster {
  int index;
  std::set<tensorflow::Node*> nodes;
};
}

Status AssignClusters(Graph* graph) {
  std::map<Node*, std::shared_ptr<Cluster>> cluster_map;

  GraphCycles gc;

  for (auto node : graph->op_nodes()) {
    int new_index = gc.NewNode();
    NGRAPH_VLOG(5) << "Creating cycle graph node: " << new_index << " for "
                   << node->name() << "[" << node->type_string() << "]";
    cluster_map[node] = std::make_shared<Cluster>();
    cluster_map[node]->index = new_index;
    cluster_map[node]->nodes.insert(node);
  }

  for (auto edge : graph->edges()) {
    Node* src = edge->src();
    Node* dst = edge->dst();

    // Skip source/sink
    if (!src->IsOp() || !dst->IsOp()) {
      continue;
    }

    // Skip NextIteration
    if (src->IsNextIteration() || dst->IsNextIteration()) {
      continue;
    }

    if (!gc.InsertEdge(cluster_map[src]->index, cluster_map[dst]->index)) {
      NGRAPH_VLOG(5) << "Failing due to cycle";
      return errors::Unimplemented(
          "Input graph has a cycle (inserting an edge from ",
          src->DebugString(), " to ", dst->DebugString(),
          " would create a cycle)");
    }
  }

  // If we wish to add a constraint that 2 particular nodes not lie in the same
  // cluster, then all we have to do is add 2 'shadow' edges and 1 'shadow' node
  // in the gc data structure between the 2 nodes. The shadow edges go from the
  // node closer to toposort source to the node closer to sink, through a shadow
  // node. src--->S--->dst. (not the other way round, else it would introduce a
  // cycle).
  // TF world node (o), gc world node (+), static input *
  // Normal edge traslation:
  // (o)---->(o)   ==>  (+)---->(+)
  // Static input edge translation:
  // (o)---->*(o)  ==>  (+)---->(+)
  //                     |       ^
  //                     |       |
  //                      --(+)--

  // The contraction only happens on 'real' edges (edges that are
  // present in the TF graph itself). Therefore the shadow edges in the gc
  // data structure will never suffer contraction. Anytime the shadow path's src
  // and dst attempt a merge (by contracting some real edge between them),
  // the shadow path will introduce a cycle and not allow it

  // Warning: this relies on the fact that we attempt to contract 'real' edges
  // from the TF graph. For optimization, one might attempt to contract the gc
  // edges, which keep decreasing unlike the TF edges. But this fix would break
  // then, since we have broken the contract that an edge in gc implies an edge
  // in TF in this fix
  for (auto node : graph->op_nodes()) {
    std::vector<int32> static_inputs;
    GetStaticInputs(node, &static_inputs);
    if (static_inputs.size() > 0) {
      std::vector<const Edge*> edges_to_node;
      TF_RETURN_IF_ERROR(node->input_edges(&edges_to_node));
      for (auto static_inp_idx : static_inputs) {
        auto static_edge = edges_to_node[static_inp_idx];
        if (static_edge->src()->type_string() != "Const") {
          int shadow_node_index = gc.NewNode();
          bool gc_success = gc.InsertEdge(
              cluster_map[static_edge->src()]->index, shadow_node_index);
          gc_success &= gc.InsertEdge(shadow_node_index,
                                      cluster_map[static_edge->dst()]->index);
          if (!gc_success)
            errors::Internal("Unable to create shadow edges in GraphCycles");
        }
      }
    }
  }

  NGRAPH_VLOG(2) << "Starting contraction";
  bool changed;

  do {
    changed = false;

    for (auto edge : graph->edges()) {
      Node* src = edge->src();
      Node* dst = edge->dst();

      if (!src->IsOp() || !dst->IsOp()) {
        continue;
      }

      int src_index = cluster_map[src]->index;
      int dst_index = cluster_map[dst]->index;

      if (!NodeIsMarkedForClustering(src) || !NodeIsMarkedForClustering(dst)) {
        NGRAPH_VLOG(5) << "Skipping (not marked): " << src->name() << "["
                       << edge->src_output() << "]@" << src_index << " -> "
                       << dst->name() << "[" << edge->dst_input() << "]@"
                       << dst_index;
        continue;
      }

      if (gc.HasEdge(src_index, dst_index) &&
          gc.ContractEdge(src_index, dst_index)) {
        NGRAPH_VLOG(5) << "Contracting: " << src->name() << "["
                       << edge->src_output() << "]@" << src_index << " -> "
                       << dst->name() << "[" << edge->dst_input() << "]@"
                       << dst_index;
        // using cluster_map[dst]->nodes in the loop directly appears to
        // invalidate the iterator when `node` == `dst`
        // this happens with clang but not gcc
        auto cluster_dst = cluster_map[dst];
        for (auto node : cluster_dst->nodes) {
          cluster_map[src]->nodes.insert(node);
          cluster_map[node] = cluster_map[src];
        }
        changed = true;
      }
    }
  } while (changed);
  NGRAPH_VLOG(2) << "Contraction done";

  NGRAPH_VLOG(2) << "Starting tagging";
  std::set<Cluster*> seen;

  for (auto kv : cluster_map) {
    auto cluster = kv.second.get();
    bool has_ngraph_ops = false;
    bool has_non_ngraph_ops = false;

    for (auto node : cluster->nodes) {
      if (NodeIsMarkedForClustering(node)) {
        has_ngraph_ops = true;
      } else {
        has_non_ngraph_ops = true;
      }
    }

    if (has_ngraph_ops && has_non_ngraph_ops) {
      NGRAPH_VLOG(2) << "Cluster " << cluster->index
                     << " has both nGraph and non-nGraph nodes";
      for (auto node : cluster->nodes) {
        NGRAPH_VLOG(2) << (NodeIsMarkedForClustering(node)
                               ? "nGraph node: "
                               : "non-nGraph node: ")
                       << node->name() << " [" << node->type_string() << "]";
      }
      return errors::Internal("Cluster ", cluster->index,
                              " has both nGraph and non-nGraph nodes");
    }

    if (!has_ngraph_ops) {
      continue;
    }

    if (seen.count(cluster) == 0) {
      int cluster_idx = NGraphClusterManager::NewCluster();

      for (auto node : cluster->nodes) {
        if (NGRAPH_VLOG_IS_ON(5)) {
          NGRAPH_VLOG(5) << ">> cluster " << cluster_idx << ": " << node
                         << " :: " << node->name() << " ["
                         << node->type_string() << "]";
        }

        if (!NodeIsMarkedForClustering(node)) {
          return errors::Internal("Node ", node->DebugString(),
                                  " was not marked for clustering but was "
                                  "placed in an nGraph cluster.");
        }

        // TODO(amprocte): move attr name to a constant
        node->AddAttr("_ngraph_cluster", cluster_idx);
      }

      seen.insert(cluster);
    }
  }
  NGRAPH_VLOG(2) << "Tagging done";

  return Status::OK();
}

Status GetNodeCluster(const Node* node, int* cluster) {
  // TODO(amprocte): move attr name to a constant
  Status s = GetNodeAttr(node->attrs(), "_ngraph_cluster", cluster);
  if (s != Status::OK()) {
    *cluster = -1;
  }
  return s;
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
