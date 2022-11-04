#pragma once

#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <random>
#include <set>
#include <thread>
#include <atomic>
#include <handlegraph/path_handle_graph.hpp>
#include <handlegraph/handle_graph.hpp>
#include "xp.hpp"
#include "IITree.h"
#include <zipfian_int_distribution.h>
#include <iomanip>
#include <string>
#include <sdsl/bit_vectors.hpp>
#include "dirty_zipfian_int_distribution.h"
#include "XoshiroCpp.hpp"
#include "progress.hpp"

namespace odgi {
    namespace algorithms {

        using namespace handlegraph;

/// use SGD driven, by path guided, and partly zipfian distribution sampled pairwise distances to obtain a 1D linear layout of the graph that respects its topology
        void path_linear_sgd_layout(const PathHandleGraph &graph,
                                    const xp::XP &path_index,
                                    const std::vector<path_handle_t> &path_sgd_use_paths,
                                    const uint64_t &iter_max,
                                    const uint64_t &iter_with_max_learning_rate,
                                    const uint64_t &min_term_updates,
                                    const double &delta,
                                    const double &eps,
                                    const double &eta_max,
                                    const double &theta,
                                    const uint64_t &space,
                                    const uint64_t &space_max,
                                    const uint64_t &space_quantization_step,
                                    const double &cooling_start,
                                    const uint64_t &nthreads,
                                    const bool &progress,
                                    const bool &snapshot,
                                    const std::string &snapshot_prefix,
                                    std::vector<std::atomic<double>> &X,
                                    std::vector<std::atomic<double>> &Y,
                                    const uint32_t &num_nodes_per_step, 
                                    const bool &all_node_combinations);

/// our learning schedule
        std::vector<double> path_linear_sgd_layout_schedule(const double &w_min,
                                                            const double &w_max,
                                                            const uint64_t &iter_max,
                                                            const uint64_t &iter_with_max_learning_rate,
                                                            const double &eps);

/// function to get "pos_in_path" and "coord_idx" for a given step
        std::tuple<size_t, uint64_t> get_pos_in_path_and_coord_idx(const PathHandleGraph &graph,
                                                                   const xp::XP &path_index, 
                                                                   std::uniform_int_distribution<uint64_t> &flip,
                                                                   XoshiroCpp::Xoshiro256Plus &gen,
                                                                   step_handle_t &step);
/// function to update coordinates X, Y; given a pair of nodes
        void update_pos(size_t &pos_in_path_a, 
                        size_t &pos_in_path_b,
                        uint64_t &coord_idx_a,
                        uint64_t &coord_idx_b,
                        std::vector<std::atomic<double>> &X,
                        std::vector<std::atomic<double>> &Y, 
                        std::atomic<double> &eta, 
                        std::atomic<double> &Delta_max);


/// single threaded and deterministic path guided 1D linear SGD
/*
        void deterministic_path_linear_sgd_layout(const PathHandleGraph &graph,
                                                  const xp::XP &path_index,
                                                  const std::vector<path_handle_t> &path_sgd_use_paths,
                                                  const uint64_t &iter_max,
                                                  const uint64_t &iter_with_max_learning_rate,
                                                  const uint64_t &min_term_updates,
                                                  const double &delta,
                                                  const double &eps,
                                                  const double &eta_max,
                                                  const double &theta,
                                                  const uint64_t &space,
                                                  const std::string &seeding_string,
                                                  const bool &progress,
                                                  const bool &snapshot,
                                                  std::vector<std::vector<double>> &snapshots,
                                                  std::vector<std::atomic<double>> &X,
                                                  std::vector<std::atomic<double>> &Y);
*/

    }

}
