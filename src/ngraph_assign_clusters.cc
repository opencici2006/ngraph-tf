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
#include "tf_deadness_analysis.h"
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
// If nodes N1, N2, N3, N4, N6, and N7 are all marked, but N5 is unmarked, the
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
// Other Constraints (Non Data Flow Constraints)
//
//   (1) If N1 is a static input to N2, N1 and N2 are not placed in the same
//       cluster (More on static inputs in ngraph_mark_for_clustering)
//   (2) If N1 and N2 have mismatching deadness predicates, they are not
//       placed in the same cluster (More on deadness in tf_deadness_analysis)
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
  std::string predicate_string;
  std::set<const Edge*> outgoing_edges;
};
}  // namespace

Status AssignClusters(Graph* graph) {
  std::map<Node*, std::shared_ptr<Cluster>> cluster_map;

  // Deadness is typically introduced by control flow ops. So, all the outgoing
  // edges from the data flow op have the same deadness predicate ('And'
  // Predicate of all its inputs) and we can attach a predicate string to the
  // data-flow node (predicate of its output edge). Control flow ops are
  // assigned a placeholder predicate string.

  // TODO (malikshr): Add FLAG to disable deadness
  // #if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
  std::unique_ptr<DeadnessAnalysis> deadness_analyzer;
  TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(*graph, &deadness_analyzer));
  std::string CONTROL_FLOW_PRED_STRING = "#control_flow";
  // Same as the True predicate in tf_deadness_analysis
  std::string TRUE_PRED_STRING = "#true";

  std::map<Node*, std::string> nodes_predicate_map;

  GraphCycles gc;

  // Initial Step: Each node is a cluster of its own
  for (auto node : graph->nodes()) {
    int new_index = gc.NewNode();
    cluster_map[node] = std::make_shared<Cluster>();

    std::string pred_string = CONTROL_FLOW_PRED_STRING;
    // if data flow op pred_string will be updated
    deadness_analyzer->GetNodePredicate(*node, pred_string);
    nodes_predicate_map[node] = pred_string;

    cluster_map[node]->index = new_index;
    cluster_map[node]->nodes.insert(node);
    cluster_map[node]->predicate_string = pred_string;

    // TODO : Try to directly create set of edges, instead of for loop
    for (const Edge* edge : node->out_edges()) {
      cluster_map[node]->outgoing_edges.insert(edge);
    }
    NGRAPH_VLOG(5) << "Creating graphcycle Node: " << new_index << " for "
                   << node->name() << "[" << node->type_string()
                   << "] Predicate : " << pred_string;
  }

  // Check for existing cyclicity in the graph
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
            return errors::Internal(
                "Unable to create shadow edges in GraphCycles");
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

      NGRAPH_VLOG(5) << "Checking Edge : " << src->name() << "["
                     << src->type_string() << " , " << edge->src_output()
                     << "]@" << src_index << " -> " << dst->name() << "["
                     << dst->type_string() << " , " << edge->dst_input() << "]@"
                     << dst_index;
      /*
      if (src_index == dst_index) {
        continue;
      }
      */
      string src_predicate = cluster_map[src]->predicate_string;
      string dst_predicate = cluster_map[dst]->predicate_string;
      NGRAPH_VLOG(5) << "Src pred: " << src_predicate
                     << " ,Dst pred: " << dst_predicate;

      if (!NodeIsMarkedForClustering(src) || !NodeIsMarkedForClustering(dst)) {
        NGRAPH_VLOG(5) << "Skipping (not marked): " << src->name() << "["
                       << edge->src_output() << "]@" << src_index << " -> "
                       << dst->name() << "[" << edge->dst_input() << "]@"
                       << dst_index;
        continue;
      }

      // If the node marked for clustering has CONTROL_FLOW_PRED_STRING, it
      // breaks our assumption that all supported ops are data flow ops, and all
      // its outputs have the same predicate
      if (src_predicate == CONTROL_FLOW_PRED_STRING ||
          dst_predicate == CONTROL_FLOW_PRED_STRING) {
        Node* err_node =
            (src_predicate == CONTROL_FLOW_PRED_STRING) ? src : dst;
        return errors::Internal(
            "Attempting to cluster node with mismatching output deadness : ",
            err_node->name(), "[", err_node->type_string(), "]");
      }

      // Case src X , dst Y , X!=Y // cannot be contracted
      if (src_predicate != TRUE_PRED_STRING &&
          dst_predicate != TRUE_PRED_STRING && src_predicate != dst_predicate) {
        continue;
      }

      // Case src X , dst True // invalid scenario
      if (src_predicate != TRUE_PRED_STRING &&
          dst_predicate == TRUE_PRED_STRING) {
        return errors::Internal("Attempting to cluster control-flow node ",
                                dst->name(), "[", dst->type_string(), "]");
      }

      // Case src True, dst Y
      // Contraction possible only when ... <TODO> Add here
      if (src_predicate == TRUE_PRED_STRING) {
        // we only care about the out preds of the src cluster as after merge
        // these edges will take the predicate of dst. No changes to the
        // outgoing edges of the dst cluster
        auto src_cluster_out_edges = cluster_map[src]->outgoing_edges;
        bool found_same_out_preds = true;
        std::string pred_check = dst_predicate;

        for (const Edge* src_cluster_edge : src_cluster_out_edges) {
          NGRAPH_VLOG(5) << " Check SRC Cluster Edge "
                         << src_cluster_edge->DebugString();
          if (src_cluster_edge == edge) {
            continue;
          }

          Node* src_cluster_dst = src_cluster_edge->dst();
          NGRAPH_VLOG(5) << " Got SRC Cluster Edge Dst";
          std::string src_cluster_dest_pred =
              cluster_map[src_cluster_dst]->predicate_string;
          NGRAPH_VLOG(5) << " Pred Check " << pred_check << " Src Out Pred "
                         << src_cluster_dest_pred;
          if (pred_check != src_cluster_dest_pred) {
            found_same_out_preds = false;
            break;
          }
        }

        // Cannot contract this edge
        if (!found_same_out_preds) {
          continue;
        }
      }

      NGRAPH_VLOG(5) << "Can Cluster";

      // Can be clustered
      // Case src True, dst True
      // Case src X, dst Y, X==Y

      // Try clustering
      if (gc.HasEdge(src_index, dst_index) &&
          gc.ContractEdge(src_index, dst_index)) {
        NGRAPH_VLOG(5) << "Contracting: " << src->name() << "["
                       << src->type_string() << " , " << edge->src_output()
                       << "]@" << src_index << " -> " << dst->name() << "["
                       << dst->type_string() << " , " << edge->dst_input()
                       << "]@" << dst_index;
        NGRAPH_VLOG(5) << "Src pred: " << src_predicate
                       << " ,Dst pred: " << dst_predicate;

        // using cluster_map[dst]->nodes in the loop directly appears to
        // invalidate the iterator when `node` == `dst`
        // this happens with clang but not gcc
        std::string cluster_pred =
            (src_predicate != TRUE_PRED_STRING) ? src_predicate : dst_predicate;
        cluster_map[src]->predicate_string = cluster_pred;
        auto cluster_dst = cluster_map[dst];

        for (auto cluster_dst_out_edge : cluster_dst->outgoing_edges) {
          cluster_map[src]->outgoing_edges.insert(cluster_dst_out_edge);
        }

        cluster_map[src]->outgoing_edges.erase(edge);

        for (auto node : cluster_dst->nodes) {
          cluster_map[src]->nodes.insert(node);
          cluster_map[node] = cluster_map[src];
        }

        // something changed
        changed = true;
      }  // try contracting
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

    // Check Deadness Clustering
    std::string cluster_pred_string = cluster->predicate_string;
    for (auto node : cluster->nodes) {
      if (nodes_predicate_map.find(node) == nodes_predicate_map.end()) {
        return errors::Internal("Node ", node->name(), " [",
                                node->type_string(), "]",
                                " not found in predicate map");
      }
      std::string node_pred_string = nodes_predicate_map[node];

      if (node_pred_string == CONTROL_FLOW_PRED_STRING) {
        return errors::Internal(
            "Node ", node->name(), " [", node->type_string(), "]",
            " should not be clustered as it is a control flow op");
      }

      if (node_pred_string != TRUE_PRED_STRING &&
          node_pred_string != cluster_pred_string) {
        return errors::Internal(
            "Node ", node->name(), " [", node->type_string(), "]",
            " Predicate : ", node_pred_string,
            "should not be clustered in cluster with pred_String ",
            cluster_pred_string);
      }
    }

    if (seen.count(cluster) == 0) {
      int cluster_idx = NGraphClusterManager::NewCluster();

      for (auto node : cluster->nodes) {
        if (NGRAPH_VLOG_IS_ON(5)) {
          NGRAPH_VLOG(5) << ">> cluster " << cluster_idx << ": " << node->id()
                         << " " << node << " :: " << node->name() << " ["
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
  // NGRAPH_VLOG(5) << "Node " << node << " : id " << node->id() << " "
  //               << node->name() << " [" << node->type_string() << "]";

  // TODO(amprocte): move attr name to a constant
  Status s = GetNodeAttr(node->attrs(), "_ngraph_cluster", cluster);
  if (s != Status::OK()) {
    *cluster = -1;
  }
  return s;
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
