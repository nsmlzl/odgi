#pragma once

#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <random>
#include <set>
#include <thread>
#include <atomic>
#include <omp.h>
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
#include "algorithms/pgsgd.h"

// #ifdef WITH_CUDA_KRNLS
// #include "cuda/layout.h"
// #endif

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
                                    std::vector<std::atomic<double>> &Y);

        void compute_layout_cpu(pgsgd::node_data_t &node_data,
                                pgsgd::path_data_t &path_data,
                                const uint64_t &nthreads,
                                double *etas,
                                double *zetas,
                                const uint64_t &min_term_updates,
                                const uint64_t &iter_max,
                                const uint64_t &first_cooling_iteration,
                                const uint64_t &space,
                                const uint64_t &space_max,
                                const uint64_t &space_quantization_step,
                                const double &theta);


        // void path_linear_sgd_layout_gpu(const PathHandleGraph &graph,
        //                             const xp::XP &path_index,
        //                             const std::vector<path_handle_t> &path_sgd_use_paths,
        //                             const uint64_t &iter_max,
        //                             const uint64_t &iter_with_max_learning_rate,
        //                             const uint64_t &min_term_updates,
        //                             const double &delta,
        //                             const double &eps,
        //                             const double &eta_max,
        //                             const double &theta,
        //                             const uint64_t &space,
        //                             const uint64_t &space_max,
        //                             const uint64_t &space_quantization_step,
        //                             const double &cooling_start,
        //                             const uint64_t &nthreads,
        //                             const bool &progress,
        //                             const bool &snapshot,
        //                             const std::string &snapshot_prefix,
        //                             std::vector<std::atomic<double>> &X,
        //                             std::vector<std::atomic<double>> &Y);

/// our learning schedule
        std::vector<double> path_linear_sgd_layout_schedule(const double &w_min,
                                                            const double &w_max,
                                                            const uint64_t &iter_max,
                                                            const uint64_t &iter_with_max_learning_rate,
                                                            const double &eps);

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
