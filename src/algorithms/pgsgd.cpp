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
                vector<odgi::path_handle_t> path_handles{};
                path_handles.reserve(graph.get_path_count());
                uint32_t total_path_steps = 0;

                graph.for_each_path_handle(
                        [&] (const odgi::path_handle_t& p) {
                        path_handles.push_back(p);
                        total_path_steps += graph.get_step_count(p);
                        });
                return total_path_steps;
            }


            uint64_t get_zeta_cnt(const uint64_t &space, const uint64_t &space_max, const uint64_t &space_quantization_step) {
                return ((space <= space_max)? space : (space_max + (space - space_max) / space_quantization_step + 1)) + 1;
            }


            void fill_pgsgd_data_structure() {
                std::cout << "hello, world!" << std::endl;
            }

        }
    }
}
