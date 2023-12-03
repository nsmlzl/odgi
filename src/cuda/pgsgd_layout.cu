#include "pgsgd_layout.h"
#include <cuda.h>

namespace odgi {
    using namespace algorithms;

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
                std::vector<std::atomic<double>> &Y) {

            std::cout << "WARNING: This software was only validated with an NVIDIA RTX A6000 GPU." << std::endl;
            std::cout << "Using different GPU models may lead to unexpected behavior or crashes." << std::endl;

            pgsgd::check_valid_graph_dim(graph);

            // create eta array
            double *etas;
            cudaMallocManaged(&etas, iter_max * sizeof(double));

            // create node data structure
            // consisting of sequence length and coords
            pgsgd::node_data_t node_data;
            node_data.node_count = graph.get_node_count();
            cudaMallocManaged(&node_data.nodes, node_data.node_count * sizeof(pgsgd::node_t));

            // create path data structure
            pgsgd::path_data_t path_data;
            path_data.path_count = graph.get_path_count();
            path_data.total_path_steps = pgsgd::get_total_path_steps(graph);
            cudaMallocManaged(&path_data.paths, path_data.path_count * sizeof(pgsgd::path_t));
            cudaMallocManaged(&path_data.element_array, path_data.total_path_steps * sizeof(pgsgd::path_element_t));

            // precomputed zetas
            double *zetas;
            cudaMallocManaged(&zetas, pgsgd::get_zeta_cnt(space, space_max, space_quantization_step) * sizeof(double));

            pgsgd::fill_pgsgd_data_structure();


            const uint64_t block_size = BLOCK_SIZE;
            uint64_t block_nbr = (min_term_updates + block_size - 1) / block_size;
            std::cout << "block_nbr: " << block_nbr << " block_size: " << block_size << std::endl;

            // rnadom states
            curandState_t *rnd_state_tmp;
            curandStateCoalesced_t *rnd_state;
            cudaError_t tmp_error = cudaMallocManaged(&rnd_state_tmp, SM_COUNT * block_size * sizeof(curandState_t));
            std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
            tmp_error = cudaMallocManaged(&rnd_state, SM_COUNT * sizeof(curandStateCoalesced_t));
            std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
            // TODO init
            cudaFree(rnd_state_tmp);

            // TODO kernel

            cudaFree(etas);
            cudaFree(node_data.nodes);
            cudaFree(path_data.paths);
            cudaFree(path_data.element_array);
            cudaFree(zetas);
            cudaFree(rnd_state);

            return;
        }
    }
}
