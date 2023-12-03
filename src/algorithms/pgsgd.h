#pragma once

#include <iostream>
#include <cstdint>
#include <math.h>
#include <vector>
#include <atomic>

#include "odgi.hpp"
#include "dirty_zipfian_int_distribution.h"

namespace odgi {
    namespace algorithms {
        namespace pgsgd {
            // struct __align__(8) node_t {
            struct node_t {
                float coords[4];
                int32_t seq_length;
            };
            struct node_data_t {
                uint32_t node_count;
                node_t *nodes;
            };


            // struct __align__(8) path_element_t {
            struct path_element_t {
                uint32_t pidx;
                uint32_t node_id;
                int64_t pos;    // if position negative: reverse orientation
            };

            struct path_t {
                uint32_t step_count;
                uint64_t first_step_in_path;  // precomputed position in path
                path_element_t *elements;
            };

            struct path_data_t {
                uint32_t path_count;
                uint32_t total_path_steps;
                path_t *paths;
                path_element_t *element_array;
            };


            void check_valid_graph_dim(const PathHandleGraph &graph);

            uint32_t get_total_path_steps(const PathHandleGraph &graph);

            uint64_t get_zeta_cnt(const uint64_t &space, const uint64_t &space_max,
                    const uint64_t &space_quantization_step);

            void fill_etas(double *etas, const int64_t &iter_max, const int64_t &iter_with_max_learning_rate,
                    const double &eps, const double &eta_max);

            void fill_node_data(pgsgd::node_data_t &node_data, const PathHandleGraph &graph,
                    std::vector<std::atomic<double>> &X, std::vector<std::atomic<double>> &Y);

            void fill_path_data(pgsgd::path_data_t &path_data, const PathHandleGraph &graph, const uint64_t &nthreads);

            void fill_zetas(double *zetas, const uint64_t &space, const uint64_t &space_max,
                    const uint64_t &space_quantization_step, const double &theta);
        }
    }
}
