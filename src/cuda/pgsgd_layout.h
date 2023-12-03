#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include "odgi.hpp"
#include "xp.hpp"
#include <handlegraph/path_handle_graph.hpp>
#include <handlegraph/handle_graph.hpp>
#include <atomic>
#include "algorithms/pgsgd.h"

#include <iostream>


#define SM_COUNT 84
#define BLOCK_SIZE 1024


namespace odgi {
    namespace cuda {
        void path_linear_sgd_layout_gpu(const handlegraph::PathHandleGraph &graph,
                const xp::XP &path_index,
                const std::vector<handlegraph::path_handle_t> &path_sgd_use_paths,
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

        struct curandStateXORWOWCoalesced_t {
            unsigned int d[BLOCK_SIZE];
            unsigned int w0[BLOCK_SIZE];
            unsigned int w1[BLOCK_SIZE];
            unsigned int w2[BLOCK_SIZE];
            unsigned int w3[BLOCK_SIZE];
            unsigned int w4[BLOCK_SIZE];
        };
        typedef struct curandStateXORWOWCoalesced_t curandStateCoalesced_t;
    }
}
