/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef SBP_CONSTRUCTOR_
#define SBP_CONSTRUCTOR_

#include "sbp_graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/mirrored_sig_infer_hint.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/operator/normal_model_update_op.h"
#include <fstream>
#include "sbp_collector.h"
#include "sbp_util.h"

namespace oneflow {

class SbpConstructor {
 public:
  // A mapping from op names to nodes in cost model of auto parallel
  HashMap<std::string, Algorithm::SbpNode<SbpSignature>*> op_name2sbp_node;
  // Time ratio for unit computation cost vs unit copy cost
  double CostRatio = -1.0;
  // Overlayable wait time for copy cost, which occurs before communication between devices.
  double wait_time = -1.0;
  // Uncovered wait time for copy cost.
  double transfer_cost = -1.0;
  // Maps operator name to the successive proxy of sbp node
  HashMap<std::string, Algorithm::SbpNode<SbpSignature>*> op_name2sbp_proxy;
  // sbp graph
  Algorithm::SbpGraph<SbpSignature> sbp_graph;

  SbpConstructor() {
    std::ifstream ifs("/home/liyipeng/OneFlow-Benchmark/Classification/cnns/CostRatioFile.txt");
    if (ifs.is_open()) {
      ifs >> CostRatio;
      ifs >> wait_time;
      ifs >> transfer_cost;
    } else {
      std::cout << "CostRatioFile.txt does not exist." << std::endl;
    }
    ifs.close();
    if (CostRatio < 0) CostRatio = 0.1;
    if (wait_time < 0) wait_time = 1.65e8;
    if (transfer_cost < 0) transfer_cost = 1.65e7;
    std::cout << "Cost Ratio: " << CostRatio << std::endl;
    std::cout << "Wait time:" << wait_time << std::endl;
    std::cout << "Transfer Cost:" << transfer_cost << std::endl;
    sbp_graph.SetWaitTime(wait_time);
    sbp_graph.SetTransferCost(transfer_cost);
  };
  ~SbpConstructor() = default;

  void constructSbpGraph(OpGraph& op_graph, Job& job);

  bool OpNodeIsMirrored(OpNode* op_node) const;

  int32_t FindAllMirroredOpNodes(HashMap<std::string, bool>& op_name2is_mirrored,
                                 OpGraph& op_graph);

  void InitializeSbpGraph(OpGraph& op_graph, HashMap<std::string, bool>& op_name2is_fixed);

  int32_t FindAllFixedOpNodes(HashMap<std::string, bool>& op_name2is_fixed, OpGraph& op_graph);

  Maybe<void> InferLogicalBlobDesc(OpGraph& op_graph, const Job& job,
                                   HashMap<std::string, bool>& op_name2is_fixed);

  void InferOpNodeSbpSignature(OpNode* op_node, const SbpSignature& sbp_sig_conf);

  Maybe<void> InferOpNodeLogicalBlobDesc(OpNode* op_node) const;

  void SplitLogicalInputBlobDesc(OpNode* op_node) const;

  // get Sbp Signature for current op
  // get the way to compute order value in this code
  Maybe<void> InferOpSbpSignature(
      const Operator& op_, const SbpSignature& sbp_sig_conf, const ParallelDesc& parallel_desc,
      const HashMap<std::string, SbpInferHint>& ibn2sbp_infer_hint,
      std::function<Maybe<const OptInt64*>(const std::string&)> BatchAxis4BnInOp);

  Maybe<void> InferSbpSignatureIf(
      const Operator& op_, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc);

  // With sbp signature fixed in upstream, determine a sbp signature for downstream
  Maybe<void> InferSbpSignature(
      const Operator& op_, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc);

  // Compute copy cost.
  void InitializeCopyCost(OpGraph& op_graph, HashMap<std::string, bool>& op_name2is_fixed);

  // Load logical blob ids onto sbp edges
  void LoadLbi2SbpEdge(OpGraph& op_graph, HashMap<std::string, bool>& op_name2is_fixed);

  // Compute computation cost for all sbp nodes
  void InitializeComputationCost(OpGraph& op_graph, HashMap<std::string, bool>& op_name2is_fixed);

  // Initialize Cost Model with Sbp from OpGraph
  void StealSbpFromOpGraph(OpGraph& op_graph, HashMap<std::string, bool>& op_name2is_fixed);

  // Update Sbp Signature in each operator
  Maybe<void> UpdateSbpSignature4Op(OpGraph& op_graph, Job& job,
                                    HashMap<std::string, bool>& op_name2is_fixed);

  // Should customize a function to compute computation cost for each kind of op
  // compute computation cost
  double ComputeComputationCost(const SbpParallel& sbp_parallel_, const BlobDesc& logical_blob_desc,
                                const ParallelDesc& parallel_desc);

  // Algorithm::SbpGraph<SbpSignature> sbp_graph;

  // Print the graph with SBP in order
  void PrintGraph(OpGraph& op_graph);
};

}  // namespace oneflow

#endif  // SBP_CONSTRUCTOR_
