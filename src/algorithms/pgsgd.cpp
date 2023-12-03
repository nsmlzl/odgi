#include "pgsgd.h"

namespace odgi {
    namespace algorithms {
        namespace pgsgd {

            void check_valid_graph_dim(const PathHandleGraph &graph) {
                uint32_t node_count = graph.get_node_count();
                // std::cout << "node_count: " << node_count << std::endl;
                assert(graph.min_node_id() == 1);
                assert(graph.max_node_id() == node_count);
                assert(graph.max_node_id() - graph.min_node_id() + 1 == node_count);
            }


            uint32_t get_total_path_steps(const PathHandleGraph &graph) {
                uint32_t total_path_steps = 0;
                graph.for_each_path_handle(
                        [&] (const odgi::path_handle_t& p) {
                            total_path_steps += graph.get_step_count(p);
                        });
                return total_path_steps;
            }


            uint64_t get_zeta_cnt(const uint64_t &space, const uint64_t &space_max,
                    const uint64_t &space_quantization_step) {
                return ((space <= space_max)? space : (space_max + (space - space_max) / space_quantization_step + 1)) + 1;
            }


            void fill_etas(double *etas, const int64_t &iter_max, const int64_t &iter_with_max_learning_rate,
                    const double &eps, const double &eta_max) {
                const double w_max = 1.0;
                const double eta_min = eps / w_max;
                const double lambda = log(eta_max / eta_min) / ((double) iter_max - 1);
                for (int32_t i = 0; i < iter_max; i++) {
                    double eta = eta_max * exp(-lambda * (std::abs(i - iter_with_max_learning_rate)));
                    etas[i] = isnan(eta)? eta_min : eta;
                }
            }


            void fill_node_data(pgsgd::node_data_t &node_data, const PathHandleGraph &graph,
                    std::vector<std::atomic<double>> &X, std::vector<std::atomic<double>> &Y) {
                for (int node_idx = 0; node_idx < node_data.node_count; node_idx++) {
                    assert(graph.has_node(node_idx));
                    pgsgd::node_t *n_tmp = &node_data.nodes[node_idx];

                    // sequence length
                    const handlegraph::handle_t h = graph.get_handle(node_idx + 1, false);
                    // NOTE: unable store orientation (reverse), since this information is path dependent
                    n_tmp->seq_length = graph.get_length(h);

                    // copy random coordinates
                    n_tmp->coords[0] = float(X[node_idx * 2].load());
                    n_tmp->coords[1] = float(Y[node_idx * 2].load());
                    n_tmp->coords[2] = float(X[node_idx * 2 + 1].load());
                    n_tmp->coords[3] = float(Y[node_idx * 2 + 1].load());
                }
            }


            void fill_path_data(pgsgd::path_data_t &path_data, const PathHandleGraph &graph, const uint64_t &nthreads) {
                // get path handles
                vector<odgi::path_handle_t> path_handles{};
                path_handles.reserve(path_data.path_count);
                graph.for_each_path_handle(
                        [&] (const odgi::path_handle_t& p) {
                            path_handles.push_back(p);
                        });

                // get length and starting position of all paths
                uint32_t first_step_counter = 0;
                for (int path_idx = 0; path_idx < path_data.path_count; path_idx++) {
                    odgi::path_handle_t p = path_handles[path_idx];
                    int step_count = graph.get_step_count(p);
                    path_data.paths[path_idx].step_count = step_count;
                    path_data.paths[path_idx].first_step_in_path = first_step_counter;
                    first_step_counter += step_count;
                }

#pragma omp parallel for num_threads(nthreads)
                for (int path_idx = 0; path_idx < path_data.path_count; path_idx++) {
                    odgi::path_handle_t p = path_handles[path_idx];
                    //std::cout << graph.get_path_name(p) << ": " << graph.get_step_count(p) << std::endl;

                    uint32_t step_count = path_data.paths[path_idx].step_count;
                    uint32_t first_step_in_path = path_data.paths[path_idx].first_step_in_path;
                    if (step_count == 0) {
                        path_data.paths[path_idx].elements = NULL;
                    } else {
                        path_element_t *cur_path = &path_data.element_array[first_step_in_path];
                        path_data.paths[path_idx].elements = cur_path;

                        odgi::step_handle_t s = graph.path_begin(p);
                        int64_t pos = 1;
                        // Iterate through path
                        for (int step_idx = 0; step_idx < step_count; step_idx++) {
                            odgi::handle_t h = graph.get_handle_of_step(s);
                            //std::cout << graph.get_id(h) << std::endl;

                            cur_path[step_idx].node_id = graph.get_id(h) - 1;
                            cur_path[step_idx].pidx = uint32_t(path_idx);
                            // store position negative when handle reverse
                            if (graph.get_is_reverse(h)) {
                                cur_path[step_idx].pos = -pos;
                            } else {
                                cur_path[step_idx].pos = pos;
                            }
                            pos += graph.get_length(h);

                            // get next step
                            if (graph.has_next_step(s)) {
                                s = graph.get_next_step(s);
                            } else if (!(step_idx == step_count-1)) {
                                // should never be reached
                                std::cout << "Error: Here should be another step" << std::endl;
                            }
                        }
                    }
                }
            }


            void fill_zetas(double *zetas, const uint64_t &space, const uint64_t &space_max,
                    const uint64_t &space_quantization_step, const double &theta) {
                double zeta_tmp = 0.0;
                for (uint64_t i = 1; i < space + 1; i++) {
                    zeta_tmp += dirtyzipf::fast_precise_pow(1.0 / i, theta);
                    if (i <= space_max) {
                        zetas[i] = zeta_tmp;
                    }
                    if (i >= space_max && (i - space_max) % space_quantization_step == 0) {
                        zetas[space_max + 1 + (i - space_max) / space_quantization_step] = zeta_tmp;
                    }
                }
            }

        }
    }
}
