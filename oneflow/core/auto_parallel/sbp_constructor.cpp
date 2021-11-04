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
#include <algorithm>
#include <type_traits>
#include <vector>
#define DEBUG_ALGORITHM_
#define TEST_DEBUG_
// #define PRINT_GRAPH_

#include "sbp_constructor.h"

using namespace Algorithm;

namespace oneflow {

namespace {

void UpdateJobParallelViewConf(
    const OpNode& op_node, const HashMap<OpBlobArg, std::vector<OpBlobArg>>& oba2sbp_identical_obas,
    JobParallelViewConf* job_parallel_view_conf) {
  auto* op_name2sbp_signature = job_parallel_view_conf->mutable_op_name2sbp_signature_conf();
  auto Update = [&](const std::string& bn) {
    const auto& sbp_parallel = op_node.sbp_signature().bn_in_op2sbp_parallel().at(bn);
    const OpBlobArg& oba = GenOpBlobArg(op_node.op().op_name(), bn);
    auto iter = oba2sbp_identical_obas.find(oba);
    if (iter == oba2sbp_identical_obas.end()) { return; }
    for (const auto& identical_obas : iter->second) {
      auto* sbp_signature = &(*op_name2sbp_signature)[identical_obas.op_name()];
      auto iter = sbp_signature->mutable_bn_in_op2sbp_parallel()->find(identical_obas.bn_in_op());
      if (iter == sbp_signature->mutable_bn_in_op2sbp_parallel()->end()) {
        CHECK(iter->second == sbp_parallel);
      } else {
        iter->second = sbp_parallel;
      }
    }
  };
  for (const auto& ibn : op_node.op().input_bns()) { Update(ibn); }
  for (const auto& obn : op_node.op().output_bns()) { Update(obn); }
}

bool IsSbpSignatureEqual(const SbpSignature& one, const SbpSignature& two) {
  return IsSbpSignatureContaining(one, two) && IsSbpSignatureContaining(two, one);
}

}  // namespace

void SbpConstructor::constructSbpGraph(OpGraph& op_graph, Job& job) {
  // Seek out mirrored parallel opnode from job parallel view;
  // JobParallelViewConf job_parallel_view_conf(job.job_parallel_view_conf());
  // Find all the mirrored op nodes and store the mapping into op_name2is_fixed.
  HashMap<std::string, bool> op_name2is_fixed;
  FindAllFixedOpNodes(op_name2is_fixed, op_graph);
  // A mapping from op names to nodes in cost model of auto parallel
  // HashMap<std::string, Algorithm::SbpNode<SbpSignature>*> op_name2sbp_node;
  // sbp graph
  // Algorithm::SbpGraph<SbpSignature> sbp_graph;
  // Should be construct only once
  CHECK(op_name2sbp_node.empty() && sbp_graph.NodeList.empty())
      << "SBP graph should only be constructed once!" << std::endl;
  // OpGraph op_graph(*job);
  InitializeSbpGraph(op_graph, op_name2is_fixed);
  // Compute layer number for each node
  int32_t max_MinLayer = sbp_graph.ComputeLayer(op_name2sbp_node);

  // Initialize sbp signature candidate list for each node
  InferLogicalBlobDesc(op_graph, job, op_name2is_fixed);

  // Compute computation cost for all sbp nodes
  InitializeComputationCost(op_graph, op_name2is_fixed);

  // Accumulate cost on the mainstem after initializing computation cost
  sbp_graph.FindMainstem(max_MinLayer, op_name2sbp_node);

#ifdef USE_SBP_COLLECTOR_
  // Load logical blobs on all sbp edges.
  LoadLbi2SbpEdge(op_graph, op_name2is_fixed);
  // Use sbp collector to create sbp proxy for nodes with multiple downstream operators.
  SbpCollector sbp_collector;
  sbp_collector.CollectUniverse(sbp_graph);
  sbp_collector.ProxySbpCandidate(op_graph, op_name2sbp_node, sbp_graph, op_name2is_fixed);
#endif  // USE_SBP_COLLECTOR_

  // Initialize copy cost
  InitializeCopyCost(op_graph, op_name2is_fixed);

  // sbp_graph.DetectAdjustOverlap(1e-3);

  // Random Initial Sbp Signatures
  sbp_graph.RandomSbpSignature();
  StealSbpFromOpGraph(op_graph, op_name2is_fixed);

  // Find proper sbp strategy
  double OrgCost = sbp_graph.ComputeCost();
  std::cout << "Initial Cost: " << OrgCost << std::endl;
  std::cout << "Elimination Number: " << sbp_graph.NodeAndEdgeEliminations() << std::endl;
  // Use greedy strategy on the shrink graph
  sbp_graph.GreedyStrategy(5);
  sbp_graph.FinalizeSbp();
  std::cout << "After searching using greedy strategy: ";
  double FinalCost = sbp_graph.ComputeCost();
  std::cout << FinalCost << std::endl;
  if (OrgCost + 1.0 < FinalCost) std::cout << "+++++ WARNING +++++" << std::endl;
  if (OrgCost > 1e+18 && FinalCost < 1.0) { std::cout << "break point here\n"; }

  std::cout << "Number of nodes: " << sbp_graph.NextId << std::endl;

  // Update Sbp Signature for each op
  UpdateSbpSignature4Op(op_graph, job, op_name2is_fixed);

#ifdef PRINT_GRAPH_
  // Now we have the sbp graph.
  // sbp_graph.PrintGraph();
  std::cout << sbp_graph.NextId << std::endl;
#endif  // PRINT_GRAPH_
}

//} // namespace oneflow

// It should be run after mirrored op is infered.
// Judge if an op node is mirrored op
bool SbpConstructor::OpNodeIsMirrored(OpNode* op_node) const {
  const auto& op_ = op_node->op();
  for (const auto& ibn : op_.input_bns()) {
    const auto& opt_mirrored_parallel = *CHECK_JUST(op_.OptMirroredParallel4BnInOp(ibn));
    if (!opt_mirrored_parallel.has_mirrored_parallel()) return false;
  }
  for (const auto& obn : op_.output_bns()) {
    const auto& opt_mirrored_parallel = *CHECK_JUST(op_.OptMirroredParallel4BnInOp(obn));
    if (!opt_mirrored_parallel.has_mirrored_parallel()) return false;
  }
  return true;
}

int32_t SbpConstructor::FindAllMirroredOpNodes(HashMap<std::string, bool>& op_name2is_mirrored,
                                               OpGraph& op_graph) {
  int32_t MirroredNumber = 0;
  op_graph.ForEachNode([&](OpNode* op_node) {
    bool is_mirrored = OpNodeIsMirrored(op_node);
    op_name2is_mirrored[op_node->op().op_name()] = is_mirrored;
    if (is_mirrored) MirroredNumber += 1;
  });
  return MirroredNumber;
}

int32_t SbpConstructor::FindAllFixedOpNodes(HashMap<std::string, bool>& op_name2is_fixed,
                                            OpGraph& op_graph) {
  int32_t FixedNumber = 0;
  FindAllMirroredOpNodes(op_name2is_fixed, op_graph);
  op_graph.ForEachNode([&](OpNode* op_node) {
    const ParallelDesc& parallel_desc = op_node->parallel_desc();
    bool is_fixed = parallel_desc.parallel_num() == 1;
    op_name2is_fixed[op_node->op().op_name()] =
        is_fixed || op_name2is_fixed[op_node->op().op_name()];
    if (is_fixed) FixedNumber += 1;
  });
  return FixedNumber;
}

void SbpConstructor::InitializeSbpGraph(OpGraph& op_graph,
                                        HashMap<std::string, bool>& op_name2is_fixed) {
  // Initialize sbp nodes
  op_graph.ForEachNode([&](OpNode* op_node) {
    // if this op node is mirrored, skip it.
    // generate sbp node in cost model and link it with corresponding op node
    if (!op_name2is_fixed[op_node->op().op_name()]) {
      Algorithm::SbpNode<SbpSignature>* sbp_node = sbp_graph.GenerateNode();
      // mapping from sbp_node to op_node
      sbp_node->op_node = op_node;
      op_name2sbp_node[op_node->op().op_name()] = sbp_node;
    }
  });
  // Initialize sbp edges
  op_graph.ForEachNode([&](OpNode* op_node) {
    // skip it if mirrored
    if (op_name2is_fixed[op_node->op().op_name()]) return;
    // get corresponding sbp node
    Algorithm::SbpNode<SbpSignature>* sbp_node = op_name2sbp_node[op_node->op().op_name()];
    for (const auto op_edge : op_node->out_edges()) {
      // don't connected with mirrored node
      const auto& end_node_name = op_edge->dst_node()->op().op_name();
      // generate sbp edge in cost model
      if (!op_name2is_fixed[end_node_name]) sbp_node->PointTo(op_name2sbp_node[end_node_name]);
    }
  });
}

// Should customize a function to compute computation cost for each kind of op
// compute computation cost
// deprecated
double SbpConstructor::ComputeComputationCost(const SbpParallel& sbp_parallel_,
                                              const BlobDesc& logical_blob_desc,
                                              const ParallelDesc& parallel_desc) {
  double total_cost = CostRatio * logical_blob_desc.shape().elem_cnt();
  // CostRatio * logical_blob_desc.shape().elem_cnt() * logical_blob_desc.shape().elem_cnt();
  if (sbp_parallel_.has_split_parallel())
    return total_cost / parallel_desc.parallel_num();
  else
    return total_cost;
}

// Compute copy cost between nodes and nodes. Skip those proxies.
void SbpConstructor::InitializeCopyCost(OpGraph& op_graph,
                                        HashMap<std::string, bool>& op_name2is_fixed) {
  // Compute copy cost for sbp edges
  op_graph.ForEachNode([&](OpNode* op_node) {
    // skip it if fixed
    if (op_name2is_fixed[op_node->op().op_name()]) return;
    // get corresponding sbp node consumer
    Algorithm::SbpNode<SbpSignature>* sbp_node_consumer = op_name2sbp_node[op_node->op().op_name()];
    // Initialize copy cost between two nodes
    for (auto* sbp_edge : sbp_node_consumer->EdgesIn) {
      // producer sbp node
      const auto* sbp_node_producer = sbp_edge->StartNode;
#ifdef USE_SBP_COLLECTOR_
      // skip it if proxy
      if (!sbp_node_producer->op_node) continue;
#endif  // USE_SBP_COLLECTOR_
      sbp_edge->Cost.resize(sbp_node_producer->SbpSignatureList.size());
      int32_t consumer_sbp_size = sbp_node_consumer->SbpSignatureList.size();
      // look through sbp signature in producer
      for (int32_t sbp_id_producer = 0;
           sbp_id_producer < sbp_node_producer->SbpSignatureList.size(); sbp_id_producer++) {
        sbp_edge->Cost[sbp_id_producer].resize(consumer_sbp_size, 0);
      }
    }
    // Find all those cases with wait time
    // Do not skip edges carrying no lbi
    sbp_node_consumer->InitializeCopyCost(false);
    for (auto* sbp_edge : sbp_node_consumer->EdgesIn) {
      // skip it if proxy
      if (!sbp_edge->StartNode->op_node) continue;
      // Reset Wait time
      for (int32_t sbp_id_producer = 0; sbp_id_producer < sbp_edge->Cost.size();
           sbp_id_producer++) {
        for (int32_t sbp_id_consumer = 0; sbp_id_consumer < sbp_edge->Cost[sbp_id_producer].size();
             sbp_id_consumer++) {
          // If transferring between devices, we need to add wait time.
          if (sbp_edge->Cost[sbp_id_producer][sbp_id_consumer] > 0.0) {
            sbp_edge->Cost[sbp_id_producer][sbp_id_consumer] = sbp_edge->WaitTime;
          }
        }
      }
      // test debug
      std::cout << sbp_edge->StartNode->op_node->op().op_name() << " to "
                << sbp_edge->EndNode->op_node->op().op_name()
                << " Wait time: " << sbp_edge->WaitTime << std::endl;
    }

    // Re-compute the costs, skip edges carrying no lbi
    sbp_node_consumer->InitializeCopyCost(true);
  });
}

// Load logical blob ids onto sbp edges
void SbpConstructor::LoadLbi2SbpEdge(OpGraph& op_graph,
                                     HashMap<std::string, bool>& op_name2is_fixed) {
  // Load logical blobs onto sbp edges
  op_graph.ForEachNode([&](OpNode* op_node) {
    // skip it if fixed
    if (op_name2is_fixed[op_node->op().op_name()]) return;
    // get corresponding sbp node consumer
    const Algorithm::SbpNode<SbpSignature>* sbp_node_consumer =
        op_name2sbp_node[op_node->op().op_name()];

    // Loading logical blobs between two nodes
    // look through input blobs
    for (const std::string& ibn : op_node->op().input_bns()) {
      // Each input blob has one source op node.
      OpNode* producer = op_node->MutSrcNode4Ibn(ibn);
      // Skip this node because it is not in SbpGraph. However, our final goal is adding every node
      // into SbpGraph.
      if (op_name2is_fixed[producer->op().op_name()]) continue;
      // producer sbp node
      const auto* sbp_node_producer = op_name2sbp_node[producer->op().op_name()];
      // TODO: recode this
      SbpEdge<SbpSignature>* edge_found =
          Algorithm::FindEdgeBetweenNodes(sbp_node_producer, sbp_node_consumer);
#ifdef USE_SBP_COLLECTOR_
      // should use assert or CHECK process here, skip for speeding up for now
      // TODO: print to error log
      // TODO: move copy cost to proxy
      if (edge_found == NULL) {
        std::cout << "producer:" << producer->op().op_name()
                  << ", out size:" << sbp_node_producer->EdgesOut.size() << std::endl;
        for (const auto* this_edge : sbp_node_producer->EdgesOut) {
          std::cout << "Out edges:" << this_edge->EndNode->op_node->op().op_name() << std::endl;
        }
        std::cout << "consumer:" << op_node->op().op_name()
                  << ", in size:" << sbp_node_consumer->EdgesIn.size() << std::endl;
        for (const auto* this_edge : sbp_node_producer->EdgesIn) {
          std::cout << "In edges:" << this_edge->StartNode->op_node->op().op_name() << std::endl;
        }
      }
#endif  // USE_SBP_COLLECTOR_
      CHECK(edge_found != NULL) << "SbpEdge not found while loading!" << std::endl;

#ifdef USE_SBP_COLLECTOR_
      // Add copy cost for each blob
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
      edge_found->LoadLbi(lbi);
#endif  // USE_SBP_COLLECTOR_
    }
  });
}

// Compute computation cost for all sbp nodes
void SbpConstructor::InitializeComputationCost(OpGraph& op_graph,
                                               HashMap<std::string, bool>& op_name2is_fixed) {
  // Compute computation cost for sbp nodes
  op_graph.ForEachNode([&](OpNode* op_node) {
    // skip it if fixed
    if (op_name2is_fixed[op_node->op().op_name()]) return;
    // get corresponding sbp node producer
    Algorithm::SbpNode<SbpSignature>* sbp_node = op_name2sbp_node[op_node->op().op_name()];
    // get parallel description. Number of devices.
    const ParallelDesc& parallel_desc = op_node->parallel_desc();

    // look through sbp signature in this sbp node
    sbp_node->Cost.resize(sbp_node->SbpSignatureList.size());
    auto logical_blob_desc4bn = [&](const std::string& bn) -> const BlobDesc& {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(bn);
      return op_node->LogicalBlobDesc4Lbi(lbi);
    };
    for (int32_t sbp_id = 0; sbp_id < sbp_node->SbpSignatureList.size(); sbp_id++) {
      sbp_node->Cost[sbp_id] =
          CostRatio
          * CHECK_JUST(op_node->op().GetComputeComplexity(sbp_node->SbpSignatureList[sbp_id],
                                                          logical_blob_desc4bn, parallel_desc));
    }
  });
}

// Initialize Cost Model with Sbp from OpGraph
void SbpConstructor::StealSbpFromOpGraph(OpGraph& op_graph,
                                         HashMap<std::string, bool>& op_name2is_fixed) {
  // Compute computation cost for sbp nodes
  op_graph.ForEachNode([&](OpNode* op_node) {
    // skip it if fixed
    if (op_name2is_fixed[op_node->op().op_name()]) return;
    // get corresponding sbp node producer
    Algorithm::SbpNode<SbpSignature>* sbp_node = op_name2sbp_node[op_node->op().op_name()];
    SbpSignature& old_one = *op_node->mut_sbp_signature();
    for (int32_t i = 0; i < sbp_node->SbpSignatureList.size(); i++) {
      SbpSignature& new_one = *sbp_node->SbpSignatureList[i];
      if (IsSbpSignatureEqual(old_one, new_one)) {
        sbp_node->FinalSbpSignatureId = i;
        break;
      }
    }
  });
}

// Update Sbp Signature in each operator
Maybe<void> SbpConstructor::UpdateSbpSignature4Op(OpGraph& op_graph, Job& job,
                                                  HashMap<std::string, bool>& op_name2is_fixed) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    // skip it if fixed
    if (op_name2is_fixed[op_node->op().op_name()]) return;
    // get corresponding sbp node producer
    Algorithm::SbpNode<SbpSignature>* sbp_node = op_name2sbp_node[op_node->op().op_name()];
    *(op_node->mut_op()->mut_sbp_signature()) = *(sbp_node->FinalSbpSignature());
    // Here we should re-infer the other parts.
    op_node->UpdateLbi2SbpParallel();
    // Update op_conf
    // TODO: add function in op_node
    op_node->mut_op()->UpdateOpconf();
  });
  // check ConstructAndInferOp in operator.cpp, might help
  // update Sbp Parallel for each blob and update them in job configure
  JobParallelViewConf* job_parallel_view_conf = job.mutable_job_parallel_view_conf();
  HashMap<OpBlobArg, std::vector<OpBlobArg>> oba2sbp_identical_obas;
  for (const auto& pair : job.helper().identical_sbp_oba_pairs().pair()) {
    oba2sbp_identical_obas[pair.first()].push_back(pair.second());
    oba2sbp_identical_obas[pair.second()].push_back(pair.first());
    // oba2sbp_identical_obas[pair.first()].push_back(pair.second());
    // oba2sbp_identical_obas[pair.second()].push_back(pair.first());
  }
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    // They should be run after we choose one sbp signature for each node
    // Is blob parallel description related to sbp-parallel of blobs? If not, just remove it
    op_node->InferBlobParallelDesc();

    /*
     * BUG: maybe get bug: indentity-layers will set different sbp_signature?
     * BUG: It may be processed in `GetEdgeCost` function
     */
    // UpdateJobParallelViewConf(*op_node, oba2sbp_identical_obas, job_parallel_view_conf);
    auto* op_name2sbp_signature = job_parallel_view_conf->mutable_op_name2sbp_signature_conf();
    (*op_name2sbp_signature)[op_node->op().op_name()] = op_node->sbp_signature();

    // Infer logical_blob_desc
    JUST(InferOpNodeLogicalBlobDesc(op_node));
    // Fill logical blob_desc signature.
    JUST(op_node->mut_op()->FillLogicalBlobDescSignature(
        [&](const std::string& bn_in_op) -> Maybe<const BlobDesc&> {
          return op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(bn_in_op));
        }));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::InferLogicalBlobDesc(OpGraph& op_graph, const Job& job,
                                                 HashMap<std::string, bool>& op_name2is_fixed) {
  JobParallelViewConf job_parallel_view_conf(job.job_parallel_view_conf());
  HashMap<OpBlobArg, std::vector<OpBlobArg>> oba2sbp_identical_obas;
  for (const auto& pair : job.helper().identical_sbp_oba_pairs().pair()) {
    oba2sbp_identical_obas[pair.first()].push_back(pair.second());
    oba2sbp_identical_obas[pair.second()].push_back(pair.first());
  }
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    // skip inferring if is fixed.
    if (op_name2is_fixed[op_node->op().op_name()]) return Maybe<void>::Ok();
    // Infer sbp_signature
    SbpSignature sbp_sig_conf;
    {
      const auto& op_name2sbp_sig_conf = job_parallel_view_conf.op_name2sbp_signature_conf();
      const auto& iter = op_name2sbp_sig_conf.find(op_node->op().op_name());
      if (iter != op_name2sbp_sig_conf.end()) { sbp_sig_conf = iter->second; }
    }
    // Assemble sbp candidates, compute cost and copy cost.
    InferOpNodeSbpSignature(op_node, sbp_sig_conf);
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::InferOpNodeLogicalBlobDesc(OpNode* op_node) const {
  auto* bn2parallel_id2blob_desc = op_node->mut_bn2parallel_id2blob_desc();
  SplitLogicalInputBlobDesc(op_node);
  int64_t parallel_num = op_node->parallel_desc().parallel_num();
  const auto& input_bns = op_node->op().input_bns();
  FOR_RANGE(int64_t, parallel_id, 0, parallel_num) {
    auto BlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
      if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
        CHECK(bn2parallel_id2blob_desc->find(bn) != bn2parallel_id2blob_desc->end());
        CHECK_EQ(bn2parallel_id2blob_desc->at(bn).size(), parallel_num);
      } else if (bn2parallel_id2blob_desc->find(bn) == bn2parallel_id2blob_desc->end()) {
        (*bn2parallel_id2blob_desc)[bn].resize(parallel_num);
      } else {
        CHECK_EQ(bn2parallel_id2blob_desc->at(bn).size(), parallel_num);
      }
      auto* blob_desc = &bn2parallel_id2blob_desc->at(bn).at(parallel_id);
      if (!*blob_desc) { blob_desc->reset(new BlobDesc(GlobalJobDesc().DefaultDataType())); }
      return blob_desc->get();
    };
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(parallel_id);
    parallel_ctx.set_parallel_num(parallel_num);
    JUST(op_node->op().InferBlobDescsIf(BlobDesc4BnInOp, &parallel_ctx, &op_node->sbp_signature(),
                                        [](OpContext*) {}));
  }
  op_node->ConcatLogicalOutputBlobDesc();
  return Maybe<void>::Ok();
}

void SbpConstructor::SplitLogicalInputBlobDesc(OpNode* op_node) const {
  for (const std::string& bn : op_node->op().input_bns()) {
    const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(bn);
    const BlobDesc& logical_blob_desc = op_node->SrcNode4Ibn(bn).LogicalBlobDesc4Lbi(lbi);
    CHECK_NE(logical_blob_desc.data_type(), DataType::kInvalidDataType);
    const SbpParallel& sbp_parallel = op_node->SbpParallel4BnInOp(bn);
    op_node->bn2parallel_id2blob_desc_[bn].resize(0);
    op_node->ForEachSplitOrBroadcastBlobDesc(
        logical_blob_desc, sbp_parallel, [&](const BlobDesc& blob_desc) {
          op_node->bn2parallel_id2blob_desc_[bn].emplace_back(new BlobDesc(blob_desc));
          CHECK_NE(op_node->bn2parallel_id2blob_desc_[bn].back()->data_type(),
                   DataType::kInvalidDataType);
        });
    CHECK_EQ(op_node->bn2parallel_id2blob_desc_.at(bn).size(),
             op_node->parallel_desc().parallel_num());
  }
}

void SbpConstructor::InferOpNodeSbpSignature(OpNode* op_node, const SbpSignature& sbp_sig_conf) {
  // assemble the mapping from input blob name to sbp infer hint in upstream of current op
  HashMap<std::string, SbpInferHint> ibn2sbp_infer_hint;
  for (const std::string& ibn : op_node->op().input_bns()) {
    const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
    // OpNode* producer = op_node->MutSrcNode4Ibn(ibn);
    const OpNode& producer = op_node->SrcNode4Ibn(ibn);
    // const OpNode& producer = op_node->ProducerOpNode4Lbi(lbi);
    const ParallelDesc* parallel_desc = &producer.parallel_desc();
    const BlobDesc* logical_blob_desc = &producer.LogicalBlobDesc4Lbi(lbi);
    const SbpParallel* sbp = &producer.SbpParallel4Lbi(lbi);
    const OptInt64* batch_axis = CHECK_JUST(producer.BatchAxis4Lbi(lbi));
    ibn2sbp_infer_hint.emplace(ibn,
                               SbpInferHint(parallel_desc, logical_blob_desc, sbp, batch_axis));
  }

  // a function mapping from blob name in operators to
  const auto& BatchAxis4BnInOp = [&](const std::string& bn_in_op) -> Maybe<const OptInt64*> {
    return op_node->op().BatchAxis4BnInOp(bn_in_op);
  };
  // Assemble sbp candidates, compute cost and copy cost.
  CHECK_JUST(InferOpSbpSignature(op_node->op(), sbp_sig_conf, op_node->parallel_desc(),
                                 ibn2sbp_infer_hint, BatchAxis4BnInOp));
}

// get Sbp Signature for current op
// get the way to compute order value in this code
Maybe<void> SbpConstructor::InferOpSbpSignature(
    const Operator& op_, const SbpSignature& sbp_sig_conf, const ParallelDesc& parallel_desc,
    const HashMap<std::string, SbpInferHint>& ibn2sbp_infer_hint,
    std::function<Maybe<const OptInt64*>(const std::string&)> BatchAxis4BnInOp) {
  // a function to get SbpInferHint with input blob name
  auto SbpInferHint4Ibn = [&](const std::string& ibn) -> Maybe<const SbpInferHint*> {
    auto it = ibn2sbp_infer_hint.find(ibn);
    if (it == ibn2sbp_infer_hint.end()) {
      return Error::CheckFailedError()
             << "cannot find corresponding SbpInferHint for input_blob_name : " << ibn;
    }
    return &(it->second);
  };
  // functions to get order value, the most important value to decide SBP
  std::function<int32_t(const SbpSignature&)> CalcOrderValue4SbpSig;
  // Give Sbp candidate highest priority if requested by user
  auto OrderValue4SbpHint = [&](const std::string& ibn,
                                const SbpParallel& sbp_parallel) -> int32_t {
    return -3 * (CHECK_JUST(SbpInferHint4Ibn(ibn))->sbp_parallel() == sbp_parallel);
  };
  // Set the function to compute order value for different Sbp Signature
  // Remove the effects from order value for now
  if (false && sbp_sig_conf.bn_in_op2sbp_parallel().empty()) {
    CalcOrderValue4SbpSig = [&](const SbpSignature& sbp_signature) -> int32_t {
      int32_t order_value = 0;
      for (const auto& ibn : op_.input_bns()) {
        const auto& sbp_parallel_it = sbp_signature.bn_in_op2sbp_parallel().find(ibn);
        CHECK(sbp_parallel_it != sbp_signature.bn_in_op2sbp_parallel().end());
        order_value += OrderValue4SbpHint(ibn, sbp_parallel_it->second);
      }
      return order_value;
    };
  } else {
    CalcOrderValue4SbpSig = [](const SbpSignature&) -> int32_t { return 0; };
  }
  // Get SBP signature for current op
  JUST(InferSbpSignatureIf(op_, sbp_sig_conf, CalcOrderValue4SbpSig, SbpInferHint4Ibn,
                           parallel_desc));
  return Maybe<void>::Ok();
}

// Get SBP signature for current op
Maybe<void> SbpConstructor::InferSbpSignatureIf(
    const Operator& op_, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) {
  // Remove this part for now.

  if (parallel_desc.parallel_num() == 1) {
    // // If we only have one machine and one device, we will choose S(0)
    // auto* bn2sbp = op_.mut_sbp_signature()->mutable_bn_in_op2sbp_parallel();
    // // should use set sbp candidates to be S(0) only.
    // for (const auto& ibn : op_.input_bns()) {
    // (*bn2sbp)[ibn].mutable_split_parallel()->set_axis(0); } for (const auto& obn :
    // op_.output_bns()) { (*bn2sbp)[obn].mutable_split_parallel()->set_axis(0); }
  } else if (parallel_desc.parallel_num() > 1) {
    // Pick the best one if we have multiple devices.
    return InferSbpSignature(op_, sbp_sig_conf, CalcOrderValue4SbpSig, SbpInferHint4Ibn,
                             parallel_desc);
  } else {
    // If no choice left, don't do anything for now.
    UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

// With sbp signature fixed in upstream, determine a sbp signature for downstream
Maybe<void> SbpConstructor::InferSbpSignature(
    const Operator& op_, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) {
  // generate sbp signatures for op and store them into sbp_sig_list
  auto LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> Maybe<const BlobDesc&> {
    const SbpInferHint* sbp_infer_hint = JUST(SbpInferHint4Ibn(ibn));
    return Maybe<const BlobDesc&>(sbp_infer_hint->logical_blob_desc());
  };
  SbpSignatureList sbp_sig_list;
  JUST(op_.GetSbpSignaturesIf(LogicalBlobDesc4Ibn, parallel_desc, &sbp_sig_list));

  // Steal sbp signature from op_graph, if op::GetSbpSignatures do not implement
  if (sbp_sig_list.sbp_signature_size() == 1) {
    *sbp_sig_list.mutable_sbp_signature(0) = *CHECK_JUST(op_.sbp_signature());
  }

  // filter out those sbp signatures who contain sbp signature configure from sbp signature list
  SbpSignatureList filtered_sbp_sigs_by_conf;
  FilterSbpSignatureList(sbp_sig_list, sbp_sig_conf, &filtered_sbp_sigs_by_conf);

  CHECK_GT_OR_RETURN(filtered_sbp_sigs_by_conf.sbp_signature_size(), 0);
  // Generate Sbp candidates for sbp node with lowest order value
  {
    SbpNode<SbpSignature>* sbp_node = op_name2sbp_node[op_.op_name()];
    int32_t sbp_signature_size = filtered_sbp_sigs_by_conf.sbp_signature_size();
    std::vector<int32_t> OrderValues(sbp_signature_size);
    for (int32_t i = 0; i < filtered_sbp_sigs_by_conf.sbp_signature_size(); i++) {
      OrderValues[i] = CalcOrderValue4SbpSig(filtered_sbp_sigs_by_conf.sbp_signature(i));
    }
    int32_t LowestOrderValue = *std::min_element(OrderValues.begin(), OrderValues.end());
    for (int32_t i = 0; i < filtered_sbp_sigs_by_conf.sbp_signature_size(); i++) {
      if (OrderValues[i] == LowestOrderValue)
        sbp_node->SbpSignatureObjList.emplace_back(filtered_sbp_sigs_by_conf.sbp_signature(i));
    }
    sbp_node->InitializeSbp();
    CHECK_GT_OR_RETURN(sbp_node->SbpSignatureList.size(), 0);
  }
  return Maybe<void>::Ok();
}

// Print the graph with SBP in order
void SbpConstructor::PrintGraph(OpGraph& op_graph) {
  // test debug
  std::cout << "Get Into Print Op Graph" << std::endl;
  // Collect op_node
  std::vector<OpNode*> NodeList;
  op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    NodeList.push_back(op_node);
    return Maybe<void>::Ok();
  });

  // test debug
  std::cout << "Deciding order" << std::endl;
  // Decide the order to vist the op
  std::vector<int32_t> order;
  Algorithm::DecideOrder(NodeList, order, [&](OpNode* a, OpNode* b) {
    return a->op().op_name().compare(b->op().op_name()) > 0;
  });

  // test debug
  std::cout << "Finish deciding order" << std::endl;

  for (int32_t i = 0; i < NodeList.size(); i++) {
    OpNode* op_node = NodeList[order[i]];
    // get corresponding sbp node
    auto it = op_name2sbp_node.find(op_node->op().op_name());
    std::cout << op_node->op().op_name() << " (^_^):" << std::endl;
    if (it != op_name2sbp_node.end()) {
      const Algorithm::SbpNode<SbpSignature>* sbp_node = it->second;
      std::cout << "Computation Cost: " << sbp_node->Cost[sbp_node->FinalSbpSignatureId];
      std::cout << ", Min Layer: " << sbp_node->MinLayer << ", Max Layer: " << sbp_node->MaxLayer
                << ", Tributary Layer: " << sbp_node->TributaryLayer
                << ", in mainstem: " << sbp_node->IfMainstem
                << ", Remain Cost: " << sbp_node->AccMainstemCost << std::endl;
    }
    // Print upstream operators
    for (const auto& ibn : op_node->op().input_bns()) {
      auto producer_node = op_node->MutSrcNode4Ibn(ibn);
      std::cout << "Pre Op:" << producer_node->op().op_name() << ": " << ibn;
      const SbpParallel& this_sbp_parallel = op_node->SbpParallel4BnInOp(ibn);
      if (this_sbp_parallel.has_split_parallel())
        std::cout << " S" << this_sbp_parallel.split_parallel().axis();
      if (this_sbp_parallel.has_broadcast_parallel()) std::cout << " B";
      if (this_sbp_parallel.has_partial_sum_parallel()) std::cout << " P";
      const auto input_blob_modifier_ = op_node->op().InputBlobModifier4Ibn(ibn);
      bool is_same_sbp = input_blob_modifier_.has_is_mutable() && input_blob_modifier_.is_mutable();
      if (is_same_sbp) std::cout << ", same SBP";
      std::cout << ", "
                << op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(ibn)).shape().elem_cnt();
      std::cout << std::endl;
      /* auto blob_desc = op_node->mut_bn2parallel_id2blob_desc()->at(ibn).at(0); */
      /* std::cout << " shape:" << blob_desc->shape().DebugStr() << std::endl; */
    }
    // Print downstream operators
    for (const auto& ibn : op_node->op().output_bns()) {
      std::cout << "Out Op:" << ibn;
      const SbpParallel& this_sbp_parallel = op_node->SbpParallel4BnInOp(ibn);
      if (this_sbp_parallel.has_split_parallel())
        std::cout << " S" << this_sbp_parallel.split_parallel().axis();
      if (this_sbp_parallel.has_broadcast_parallel()) std::cout << " B";
      if (this_sbp_parallel.has_partial_sum_parallel()) std::cout << " P";
      std::cout << ", "
                << op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(ibn)).shape().elem_cnt();
      std::cout << std::endl;
      /* auto blob_desc = op_node->mut_bn2parallel_id2blob_desc()->at(ibn).at(0); */
      /* std::cout << " shape:" << blob_desc->shape().DebugStr() << std::endl; */
    }
    // Print all the enforced upstream operators.
    for (const auto& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      std::cout << "Ctrl In Op: " << ctrl_in_op_name << std::endl;
    }
  }
}

}  // namespace oneflow
