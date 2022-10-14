#include "rnd_node_generator.h"

namespace python_extension {

    RndNodeGenerator::RndNodeGenerator(odgi::graph_t &graph, double zipf_theta, uint64_t space_max, uint64_t space_quantization_step) : _graph{graph} {
        this->_path_index.from_handle_graph(this->_graph, 1);

        this->_theta = zipf_theta;
        uint64_t max_path_step_count = 0;
        this->_graph.for_each_path_handle(
                [&] (const handlegraph::path_handle_t &path) {
                    max_path_step_count = std::max(max_path_step_count, this->_path_index.get_path_step_count(path));
                });
        this->_space = max_path_step_count;
        this->_space_max = space_max;
        this->_space_quantization_step = space_quantization_step;

        // implemented as in path_sgd_layout.cpp
        this->_zetas = std::vector<double>((this->_space <= this->_space_max ? this->_space : this->_space_max + (this->_space - this->_space_max) / this->_space_quantization_step + 1)+1);
        uint64_t last_quantized_i = 0;
        for (uint64_t i = 1; i < this->_space+1; ++i) {
            uint64_t quantized_i = i;
            uint64_t compressed_space = i;
            if (i > this->_space_max){
                quantized_i = this->_space_max + (i - this->_space_max) / this->_space_quantization_step + 1;
                compressed_space = this->_space_max + ((i - this->_space_max) / this->_space_quantization_step) * this->_space_quantization_step;
            }

            if (quantized_i != last_quantized_i){
                dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, compressed_space, this->_theta);
                this->_zetas[quantized_i] = z_p.zeta();
                last_quantized_i = quantized_i;
            }
        }

        this->seed = 9399220;
    }


    RndNodeGenerator::~RndNodeGenerator() {
    }

    random_nodes_pack_t RndNodeGenerator::get_random_node_pack(bool cooling) {
        int64_t id_i;
        int64_t id_j;
        int64_t vis_i;
        int64_t vis_j;
        double distance;
        // TODO fix
        //this->get_random_node_batch(1, &id_i, &id_j, &vis_i, &vis_j, &distance, cooling, 1);

        random_nodes_pack_t pack;
        pack.id_n0 = id_i;
        pack.id_n1 = id_j;
        pack.vis_p_n0 = vis_i;
        pack.vis_p_n1 = vis_j;
        pack.distance = distance;
        return pack;
    }

    void RndNodeGenerator::get_random_node_batch(int batch_size, int64_t *vis_i, int64_t *vis_j, double *d, bool cooling, int nthreads) {
        const uint64_t space = this->_space;
        const uint64_t space_max = this->_space_max;
        const uint64_t space_quantization_step = this->_space_quantization_step;
        const double theta = this->_theta;

        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            // everyone tries to seed with their own random data
            const std::uint64_t local_seed = this->seed++;
            XoshiroCpp::Xoshiro256Plus gen(local_seed); // a nice, fast PRNG
            // some references to literal bitvectors in the path index hmmm
            const sdsl::bit_vector &np_bv = this->_path_index.get_np_bv();
            const sdsl::int_vector<> &nr_iv = this->_path_index.get_nr_iv();
            const sdsl::int_vector<> &npi_iv = this->_path_index.get_npi_iv();
            // we'll sample from all path steps
            std::uniform_int_distribution<uint64_t> dis_step = std::uniform_int_distribution<uint64_t>(0, np_bv.size() - 1);
            std::uniform_int_distribution<uint64_t> flip(0, 1);

            int steps_per_thread = batch_size / nthreads + 1;
            int start_idx = tid * steps_per_thread;
            int end_idx = (start_idx+steps_per_thread <= batch_size)? start_idx + steps_per_thread : batch_size;
            for (int idx = start_idx; idx < end_idx; idx++) {
                // sample the first node from all the nodes in the graph
                // pick a random position from all paths
                uint64_t step_index = dis_step(gen);
                uint64_t path_i = npi_iv[step_index];
                handlegraph::path_handle_t path = handlegraph::as_path_handle(path_i);
                handlegraph::step_handle_t step_a, step_b;
                as_integers(step_a)[0] = path_i; // path index
                size_t s_rank = nr_iv[step_index] - 1; // step rank in path
                as_integers(step_a)[1] = s_rank;
                size_t path_step_count = this->_path_index.get_path_step_count(path);
                if (path_step_count == 1){
                    continue;
                }

                if (cooling || flip(gen)) {
                    if (s_rank > 0 && flip(gen) || s_rank == path_step_count-1) {
                        // go backward
                        uint64_t jump_space = std::min(space, s_rank);
                        uint64_t space = jump_space;
                        if (jump_space > space_max){
                            space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                        }
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, theta, this->_zetas[space]);
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                        uint64_t z_i = z(gen);
                        //assert(z_i <= path_space);
                        as_integers(step_b)[0] = as_integer(path);
                        as_integers(step_b)[1] = s_rank - z_i;
                    } else {
                        // go forward
                        uint64_t jump_space = std::min(space, path_step_count - s_rank - 1);
                        uint64_t space = jump_space;
                        if (jump_space > space_max){
                            space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                        }
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, theta, this->_zetas[space]);
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                        uint64_t z_i = z(gen);
                        //assert(z_i <= path_space);
                        as_integers(step_b)[0] = as_integer(path);
                        as_integers(step_b)[1] = s_rank + z_i;
                    }
                } else {
                    // sample randomly across the path
                    std::uniform_int_distribution<uint64_t> rando(0, path_step_count-1);
                    as_integers(step_b)[0] = as_integer(path);
                    as_integers(step_b)[1] = rando(gen);
                }


                // and the graph handles, which we need to record the update
                handlegraph::handle_t term_i = this->_path_index.get_handle_of_step(step_a);
                handlegraph::handle_t term_j = this->_path_index.get_handle_of_step(step_b);
                uint64_t term_i_length = this->_graph.get_length_no_locking(term_i);
                uint64_t term_j_length = this->_graph.get_length_no_locking(term_j);

                uint64_t id_n0 = this->_graph.get_id(term_i);
                uint64_t id_n1 = this->_graph.get_id(term_j);

                // adjust the positions to the node starts
                size_t pos_in_path_a = this->_path_index.get_position_of_step(step_a);
                size_t pos_in_path_b = this->_path_index.get_position_of_step(step_b);

                // determine which end we're working with for each node
                bool term_i_is_rev = this->_graph.get_is_reverse(term_i);
                bool use_other_end_a = flip(gen); // 1 == +; 0 == -
                if (use_other_end_a) {
                    pos_in_path_a += term_i_length;
                    // flip back if we were already reversed
                    use_other_end_a = !term_i_is_rev;
                } else {
                    use_other_end_a = term_i_is_rev;
                }
                bool term_j_is_rev = this->_graph.get_is_reverse(term_j);
                bool use_other_end_b = flip(gen); // 1 == +; 0 == -
                if (use_other_end_b) {
                    pos_in_path_b += term_j_length;
                    // flip back if we were already reversed
                    use_other_end_b = !term_j_is_rev;
                } else {
                    use_other_end_b = term_j_is_rev;
                }

                // establish the term distance
                double term_dist = std::abs(
                        static_cast<double>(pos_in_path_a) - static_cast<double>(pos_in_path_b));

                if (term_dist == 0.0) {
                    term_dist = 1e-9;
                }

                vis_i[idx] = use_other_end_a? (id_n0-1)*2 + 1 : (id_n0-1)*2;
                vis_j[idx] = use_other_end_b? (id_n1-1)*2 + 1 : (id_n1-1)*2;
                d[idx] = term_dist;

            }
        }
    }

    uint64_t RndNodeGenerator::get_max_path_length(void) {
        return this->_space;
    }
}
