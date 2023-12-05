#include "pgsgd_layout.h"
#include <cuda.h>

namespace odgi {
    using namespace algorithms;

    namespace cuda {


        __global__ void cuda_device_init(curandState_t *rnd_state_tmp, curandStateCoalesced_t *rnd_state) {
            int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            // initialize curandState with original curand implementation
            curand_init(42+tid, tid, 0, &rnd_state_tmp[tid]);
            // copy to coalesced data structure
            rnd_state[blockIdx.x].d[threadIdx.x] = rnd_state_tmp[tid].d;
            rnd_state[blockIdx.x].w0[threadIdx.x] = rnd_state_tmp[tid].v[0];
            rnd_state[blockIdx.x].w1[threadIdx.x] = rnd_state_tmp[tid].v[1];
            rnd_state[blockIdx.x].w2[threadIdx.x] = rnd_state_tmp[tid].v[2];
            rnd_state[blockIdx.x].w3[threadIdx.x] = rnd_state_tmp[tid].v[3];
            rnd_state[blockIdx.x].w4[threadIdx.x] = rnd_state_tmp[tid].v[4];
        }


        __device__ float curand_uniform_coalesced(curandStateCoalesced_t *state, uint32_t thread_id) {
            // generate 32 bit pseudorandom value with XORWOW generator (see paper "Xorshift RNGs" by George Marsaglia);
            // also used in curand library (see curand_kernel.h)
            uint32_t t;
            t = state->w0[thread_id] ^ (state->w0[thread_id] >> 2);
            state->w0[thread_id] = state->w1[thread_id];
            state->w1[thread_id] = state->w2[thread_id];
            state->w2[thread_id] = state->w3[thread_id];
            state->w3[thread_id] = state->w4[thread_id];
            state->w4[thread_id] = (state->w4[thread_id] ^ (state->w4[thread_id] << 4)) ^ (t ^ (t << 1));
            state->d[thread_id] += 362437;

            uint32_t rnd_uint = state->d[thread_id] + state->w4[thread_id];

            // convert to float; see curand_uniform.h
            return _curand_uniform(rnd_uint);
        }


        // this function uses the cuda operation __powf, which is a faster but less precise alternative to the pow operation
        __device__ uint32_t cuda_rnd_zipf(curandStateCoalesced_t *rnd_state, uint32_t n, double theta, double zeta2, double zetan) {
            double alpha = 1.0 / (1.0 - theta);
            double denominator = 1.0 - zeta2 / zetan;
            if (denominator == 0.0) {
                denominator = 1e-9;
            }
            double eta = (1.0 - __powf(2.0 / double(n), 1.0 - theta)) / (denominator);

            // INFO: curand_uniform generates random values between 0.0 (excluded) and 1.0 (included)
            double u = 1.0 - curand_uniform_coalesced(rnd_state, threadIdx.x);
            double uz = u * zetan;

            int64_t val = 0;
            if (uz < 1.0) val = 1;
            else if (uz < 1.0 + __powf(0.5, theta)) val = 2;
            else val = 1 + int64_t(double(n) * __powf(eta * u - eta + 1.0, alpha));

            if (val > n) {
                //printf("WARNING: val: %ld, n: %u\n", val, uint32_t(n));
                val--;
            }
            assert(val >= 0);
            assert(val <= n);
            return uint32_t(val);
        }


        static __device__ __inline__ uint32_t __mysmid(){
            uint32_t smid;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
            return smid;
        }


        __global__ void cuda_device_layout(pgsgd::node_data_t node_data,
                                           pgsgd::path_data_t path_data,
                                           curandStateCoalesced_t *rnd_state,
                                           int iter,
                                           double eta,
                                           double *zetas,
                                           const uint64_t first_cooling_iteration,
                                           const uint64_t space,
                                           const uint64_t space_max,
                                           const uint64_t space_quantization_step,
                                           const double theta) {
            uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            uint32_t smid = __mysmid();
            assert(smid < 84);
            curandStateCoalesced_t *thread_rnd_state = &rnd_state[smid];

            __shared__ bool cooling[32];
            if (threadIdx.x % 32 == 1) {
                cooling[threadIdx.x / 32] = (iter >= first_cooling_iteration) || (curand_uniform_coalesced(thread_rnd_state, threadIdx.x) <= 0.5);
            }

            // select path
            __shared__ uint32_t first_step_idx[32];
            if (threadIdx.x % 32 == 0) {
                // INFO: curand_uniform generates random values between 0.0 (excluded) and 1.0 (included)
                first_step_idx[threadIdx.x / 32] = uint32_t(floor((1.0 - curand_uniform_coalesced(thread_rnd_state, threadIdx.x)) * float(path_data.total_path_steps)));
                assert(first_step_idx[threadIdx.x / 32] < path_data.total_path_steps);
            }
            __syncwarp();

            // find path of step of specific thread with LUT (threads in warp pick same path)
            uint32_t step_idx = first_step_idx[threadIdx.x / 32];
            uint32_t path_idx = path_data.element_array[step_idx].pidx;


            pgsgd::path_t p = path_data.paths[path_idx];
            if (p.step_count < 2) {
                return;
            }
            assert(p.step_count > 1);

            // INFO: curand_uniform generates random values between 0.0 (excluded) and 1.0 (included)
            uint32_t s1_idx = uint32_t(floor((1.0 - curand_uniform_coalesced(thread_rnd_state, threadIdx.x)) * float(p.step_count)));
            assert(s1_idx < p.step_count);
            uint32_t s2_idx;

            if (cooling[threadIdx.x / 32]) {
                bool backward;
                uint32_t jump_space;
                if (s1_idx > 0 && (curand_uniform_coalesced(thread_rnd_state, threadIdx.x) <= 0.5) || s1_idx == p.step_count-1) {
                    // go backward
                    backward = true;
                    jump_space = min(uint32_t(space), s1_idx);
                } else {
                    // go forward
                    backward = false;
                    jump_space = min(uint32_t(space), p.step_count - s1_idx - 1);
                }
                uint32_t space = jump_space;
                if (jump_space > space_max) {
                    space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                }

                uint32_t z_i = cuda_rnd_zipf(thread_rnd_state, jump_space, theta, zetas[2], zetas[space]);

                /*
                   if (backward) {
                   if (!(z_i <= s1_idx)) {
                   printf("Error (thread %i): %u - %u\n", threadIdx.x, s1_idx, z_i);
                   printf("Jumpspace %u, theta %f, zeta %f\n", jump_space, theta, zetas[space]);
                   }
                   assert(z_i <= s1_idx);
                   } else {
                   if (!(z_i <= p.step_count - s1_idx - 1)) {
                   printf("Error (thread %i): %u + %u, step_count %u\n", threadIdx.x, s1_idx, z_i, p.step_count);
                   printf("Jumpspace %u, theta %f, zeta %f\n", jump_space, theta, zetas[space]);
                   }
                   assert(s1_idx + z_i < p.step_count);
                   }
                 */

                s2_idx = backward? s1_idx - z_i: s1_idx + z_i;
            } else {
                do {
                    s2_idx = uint32_t(floor((1.0 - curand_uniform_coalesced(thread_rnd_state, threadIdx.x)) * float(p.step_count)));
                } while (s1_idx == s2_idx);
            }
            assert(s1_idx < p.step_count);
            assert(s2_idx < p.step_count);
            assert(s1_idx != s2_idx);


            uint32_t n1_id = p.elements[s1_idx].node_id;
            int64_t n1_pos_in_path = p.elements[s1_idx].pos;
            bool n1_is_rev = (n1_pos_in_path < 0)? true: false;
            n1_pos_in_path = std::abs(n1_pos_in_path);

            uint32_t n2_id = p.elements[s2_idx].node_id;
            int64_t n2_pos_in_path = p.elements[s2_idx].pos;
            bool n2_is_rev = (n2_pos_in_path < 0)? true: false;
            n2_pos_in_path = std::abs(n2_pos_in_path);

            uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length;
            bool n1_use_other_end = (curand_uniform_coalesced(thread_rnd_state, threadIdx.x) <= 0.5)? true: false;
            if (n1_use_other_end) {
                n1_pos_in_path += uint64_t{n1_seq_length};
                n1_use_other_end = !n1_is_rev;
            } else {
                n1_use_other_end = n1_is_rev;
            }

            uint32_t n2_seq_length = node_data.nodes[n2_id].seq_length;
            bool n2_use_other_end = (curand_uniform_coalesced(thread_rnd_state, threadIdx.x) <= 0.5)? true: false;
            if (n2_use_other_end) {
                n2_pos_in_path += uint64_t{n2_seq_length};
                n2_use_other_end = !n2_is_rev;
            } else {
                n2_use_other_end = n2_is_rev;
            }

            double term_dist = std::abs(static_cast<double>(n1_pos_in_path) - static_cast<double>(n2_pos_in_path));

            if (term_dist < 1e-9) {
                term_dist = 1e-9;
            }

            double w_ij = 1.0 / term_dist;

            double mu = eta * w_ij;
            if (mu > 1.0) {
                mu = 1.0;
            }

            double d_ij = term_dist;

            int n1_offset = n1_use_other_end? 2: 0;
            int n2_offset = n2_use_other_end? 2: 0;

            float *x1 = &node_data.nodes[n1_id].coords[n1_offset];
            float *x2 = &node_data.nodes[n2_id].coords[n2_offset];
            float *y1 = &node_data.nodes[n1_id].coords[n1_offset + 1];
            float *y2 = &node_data.nodes[n2_id].coords[n2_offset + 1];
            double x1_val = double(*x1);
            double x2_val = double(*x2);
            double y1_val = double(*y1);
            double y2_val = double(*y2);

            double dx = x1_val - x2_val;
            double dy = y1_val - y2_val;

            if (dx == 0.0) {
                dx = 1e-9;
            }

            double mag = sqrt(dx * dx + dy * dy);
            double delta = mu * (mag - d_ij) / 2.0;
            //double delta_abs = std::abs(delta);

            double r = delta / mag;
            double r_x = r * dx;
            double r_y = r * dy;
            atomicExch(x1, float(x1_val - r_x));
            atomicExch(x2, float(x2_val + r_x));
            atomicExch(y1, float(y1_val - r_y));
            atomicExch(y2, float(y2_val + r_y));
        }


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

            std::cout << "WARNING: This software was only tested with an NVIDIA RTX A6000 GPU." << std::endl;
            std::cout << "Using different GPU models may lead to unexpected behavior or crashes." << std::endl;

            pgsgd::check_valid_graph_dim(graph);
            const uint64_t first_cooling_iteration = std::floor(cooling_start * (double)iter_max);

            // create eta array
            double *etas;
            cudaMallocManaged(&etas, iter_max * sizeof(double));
            pgsgd::fill_etas(etas, iter_max, iter_with_max_learning_rate, eps, eta_max);

            // create node data structure
            // consisting of sequence length and coords
            pgsgd::node_data_t node_data;
            node_data.node_count = graph.get_node_count();
            cudaMallocManaged(&node_data.nodes, node_data.node_count * sizeof(pgsgd::node_t));
            pgsgd::fill_node_data(node_data, graph, X, Y);

            // create path data structure
            pgsgd::path_data_t path_data;
            path_data.path_count = graph.get_path_count();
            path_data.total_path_steps = pgsgd::get_total_path_steps(graph);
            cudaMallocManaged(&path_data.paths, path_data.path_count * sizeof(pgsgd::path_t));
            cudaMallocManaged(&path_data.element_array, path_data.total_path_steps * sizeof(pgsgd::path_element_t));
            pgsgd::fill_path_data(path_data, graph, nthreads);

            // precomputed zetas
            double *zetas;
            cudaMallocManaged(&zetas, pgsgd::get_zeta_cnt(space, space_max, space_quantization_step) * sizeof(double));
            pgsgd::fill_zetas(zetas, space, space_max, space_quantization_step, theta);


            const uint64_t block_size = BLOCK_SIZE;
            uint64_t block_nbr = (min_term_updates + block_size - 1) / block_size;
            std::cout << "block_nbr: " << block_nbr << " block_size: " << block_size << std::endl;

            // random states
            curandState_t *rnd_state_tmp;
            curandStateCoalesced_t *rnd_state;
            cudaError_t tmp_error = cudaMallocManaged(&rnd_state_tmp, SM_COUNT * block_size * sizeof(curandState_t));
            std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
            tmp_error = cudaMallocManaged(&rnd_state, SM_COUNT * sizeof(curandStateCoalesced_t));
            std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;

            // init random states and copy to coalesced structure
            cuda_device_init<<<SM_COUNT, block_size>>>(rnd_state_tmp, rnd_state);
            tmp_error = cudaDeviceSynchronize();
            std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
            cudaFree(rnd_state_tmp);


            // compute kernel
            for (int iter = 0; iter < iter_max; iter++) {
                cuda_device_layout<<<block_nbr, block_size>>>(node_data,
                        path_data, rnd_state, iter, etas[iter], zetas,
                        first_cooling_iteration, space, space_max,
                        space_quantization_step, theta);
                cudaError_t error = cudaDeviceSynchronize();
                std::cout << "CUDA Error: " << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
            }


            pgsgd::copy_node_coords(node_data, X, Y);

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
