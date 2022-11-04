#include "path_sgd_layout.hpp"
#include "algorithms/layout.hpp"
#include <tuple>
// #define debug_sample_from_nodes
#define increase_compute
// #define more_pairs
// #define debug_test
// #define debug_segfault

namespace odgi {
    namespace algorithms {

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
                                    const bool &all_node_combinations) {

            std::cerr << "iter_max: " << iter_max << std::endl;
            std::cerr << "min_term_updates: " << min_term_updates << std::endl;

#ifdef increase_compute            
            // uint32_t num_nodes_per_step;
            // num_nodes_per_step = 4;
            std::cerr << "num_nodes_per_step: " << num_nodes_per_step << std::endl;
            std::cerr << "all_node_combinations: " << all_node_combinations << std::endl;
#endif

#ifdef debug_path_sgd
            std::cerr << "iter_max: " << iter_max << std::endl;
            std::cerr << "min_term_updates: " << min_term_updates << std::endl;
            std::cerr << "delta: " << delta << std::endl;
            std::cerr << "eps: " << eps << std::endl;
            std::cerr << "theta: " << theta << std::endl;
            std::cerr << "space: " << space << std::endl;
            std::cerr << "space_max: " << space_max << std::endl;
            std::cerr << "space_quantization_step: " << space_quantization_step << std::endl;
            std::cerr << "cooling_start: " << cooling_start << std::endl;
#endif

            uint64_t first_cooling_iteration = std::floor(cooling_start * (double)iter_max);
            //std::cerr << "first cooling iteration " << first_cooling_iteration << std::endl;

            uint64_t total_term_updates = iter_max * min_term_updates;
            std::unique_ptr<progress_meter::ProgressMeter> progress_meter;
            if (progress) {
                progress_meter = std::make_unique<progress_meter::ProgressMeter>(
                    total_term_updates, "[odgi::path_linear_sgd_layout] 2D path-guided SGD:");
            }
            using namespace std::chrono_literals; // for timing stuff
            uint64_t num_nodes = graph.get_node_count();
            // is a snapshot in progress?
            atomic<bool> snapshot_in_progress;
            snapshot_in_progress.store(false);
            // here we record which snapshots were already processed
            std::vector<atomic<bool>> snapshot_progress(iter_max);
            // we will produce one less snapshot compared to iterations
            snapshot_progress[0].store(true);
            // seed them with the graph order
            uint64_t len = 0;
            // the longest path length measured in nucleotides
            size_t longest_path_in_nucleotides = 0;
            // the total path length in nucleotides
            size_t total_path_len_in_nucleotides = 0;

            bool at_least_one_path_with_more_than_one_step = false;

            for (auto &path : path_sgd_use_paths) {
                if (path_index.get_path_step_count(path) > 1){
                    at_least_one_path_with_more_than_one_step = true;
                    break;
                }
            }


            if (at_least_one_path_with_more_than_one_step){
                double w_min = (double) 1.0 / (double) (eta_max);

#ifdef debug_path_sgd
                std::cerr << "w_min " << w_min << std::endl;
#endif
                double w_max = 1.0;
                // get our schedule
                std::vector<double> etas = path_linear_sgd_layout_schedule(w_min, w_max, iter_max,
                                                                           iter_with_max_learning_rate,
                                                                           eps);

                // cache zipf zetas for our full path space (heavy, but one-off)
                std::vector<double> zetas((space <= space_max ? space : space_max + (space - space_max) / space_quantization_step + 1)+1);
                uint64_t last_quantized_i = 0;
#pragma omp parallel for schedule(static,1)
                for (uint64_t i = 1; i < space+1; ++i) {
                    uint64_t quantized_i = i;
                    uint64_t compressed_space = i;
                    if (i > space_max){
                        quantized_i = space_max + (i - space_max) / space_quantization_step + 1;
                        compressed_space = space_max + ((i - space_max) / space_quantization_step) * space_quantization_step;
                    }

                    if (quantized_i != last_quantized_i){
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, compressed_space, theta);
                        zetas[quantized_i] = z_p.zeta();

                        last_quantized_i = quantized_i;
                    }
                }

                // how many term updates we make
                std::atomic<uint64_t> term_updates;
                term_updates.store(0);
                // learning rate
                std::atomic<double> eta;
                eta.store(etas.front());
                // adaptive zip theta
                std::atomic<double> adj_theta;
                adj_theta.store(theta);
                // if we're in a final cooling phase (last 10%) of iterations
                std::atomic<bool> cooling;
                cooling.store(false);
                // our max delta
                std::atomic<double> Delta_max;
                Delta_max.store(0);
                // should we keep working?
                std::atomic<bool> work_todo;
                work_todo.store(true);
                // approximately what iteration we're on
                uint64_t iteration = 0;
                // launch a thread to update the learning rate, count iterations, and decide when to stop
                auto checker_lambda =
                        [&]() {
                            while (work_todo.load()) {
                                if (term_updates.load() > min_term_updates) {
                                    if (snapshot) {
                                        if (snapshot_progress[iteration].load() || iteration == iter_max) {
                                            iteration++;
                                            if (iteration == iter_max) {
                                                snapshot_in_progress.store(false);
                                            } else {
                                                snapshot_in_progress.store(true);
                                            }
                                        } else {
                                            snapshot_in_progress.store(true);
                                            continue;
                                        }
                                    } else {
                                        iteration++;
                                        snapshot_in_progress.store(false);
                                    }
                                    if (iteration > iter_max) {
                                        work_todo.store(false);
                                    } else if (Delta_max.load() <= delta) { // nb: this will also break at 0
                                        if (progress) {
                                            std::cerr << "[odgi::path_linear_sgd_layout] delta_max: " << Delta_max.load()
                                                      << " <= delta: "
                                                      << delta << ". Threshold reached, therefore ending iterations."
                                                      << std::endl;
                                        }
                                        work_todo.store(false);
                                    } else {
                                        eta.store(etas[iteration]); // update our learning rate
                                        Delta_max.store(delta); // set our delta max to the threshold
                                        if (iteration > first_cooling_iteration) {
                                            //std::cerr << std::endl << "setting cooling!!" << std::endl;
                                            adj_theta.store(0.001);
                                            cooling.store(true);
                                        }
                                    }
                                    term_updates.store(0);
                                }
                                std::this_thread::sleep_for(1ms);
                            }
                        };

                auto worker_lambda =
                        [&](uint64_t tid) {
                            // everyone tries to seed with their own random data
                            const std::uint64_t seed = 9399220 + tid;
                            XoshiroCpp::Xoshiro256Plus gen(seed); // a nice, fast PRNG
                            // some references to literal bitvectors in the path index hmmm
                            const sdsl::bit_vector &np_bv = path_index.get_np_bv();
                            const sdsl::int_vector<> &nr_iv = path_index.get_nr_iv();
                            const sdsl::int_vector<> &npi_iv = path_index.get_npi_iv();
                            // we'll sample from all path steps
                            std::uniform_int_distribution<uint64_t> dis_step = std::uniform_int_distribution<uint64_t>(0, np_bv.size() - 1);
                            std::uniform_int_distribution<uint64_t> flip(0, 1);
                            uint64_t steps_since_term_updates = 0;
                            while (work_todo.load()) {
                                if (!snapshot_in_progress.load()) {
                                    // sample the first node from all the nodes in the graph
                                    // pick a random position from all paths
                                    uint64_t step_index = dis_step(gen);
#ifdef debug_sample_from_nodes
                                    std::cerr << "step_index: " << step_index << std::endl;
#endif
                                    uint64_t path_i = npi_iv[step_index];
                                    path_handle_t path = as_path_handle(path_i);

                                    size_t path_step_count = path_index.get_path_step_count(path);
                                    if (path_step_count == 1){
                                        continue;
                                    }

#ifdef debug_sample_from_nodes
                                    std::cerr << "path integer: " << path_i << std::endl;
#endif
                                    step_handle_t step_a, step_b;
                                    as_integers(step_a)[0] = path_i; // path index
                                    size_t s_rank = nr_iv[step_index] - 1; // step rank in path
                                    as_integers(step_a)[1] = s_rank;
#ifdef debug_sample_from_nodes
                                    std::cerr << "step rank in path: " << nr_iv[step_index]  << std::endl;
#endif

#ifdef increase_compute
                                    // step_handle vector
                                    std::vector<step_handle_t> steps_in_path;
                                    steps_in_path.reserve(num_nodes_per_step);
                                    steps_in_path.emplace_back(step_a);
#ifdef debug_segfault
                                    std::cerr << "size of steps_in_path: " << steps_in_path.size() << std::endl;
#endif
                                    step_handle_t step_tmp;
                                    // step_handle_t step_c, step_d; // get 4 nodes in one step. 
#endif
                                    if (cooling.load() || flip(gen)) {
                                        if (s_rank > 0 && flip(gen) || s_rank == path_step_count-1) {
                                            // go backward
                                            uint64_t jump_space = std::min(space, s_rank);
                                            uint64_t space = jump_space;
                                            if (jump_space > space_max){
                                                space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                                            }
                                            dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, theta, zetas[space]);
                                            dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                                            uint64_t z_i = z(gen);
                                            //assert(z_i <= path_space);
                                            as_integers(step_b)[0] = as_integer(path);
                                            as_integers(step_b)[1] = s_rank - z_i;

#ifdef increase_compute
                                            steps_in_path.emplace_back(step_b);
                                            for (uint64_t i = 2; i < num_nodes_per_step; ++i) {
                                                // Choose more nodes from the same path in each "step"
                                                z_i = z(gen);
                                                as_integers(step_tmp)[0] = as_integer(path);
                                                as_integers(step_tmp)[1] = s_rank - z_i;
                                                steps_in_path.emplace_back(step_tmp);
#ifdef debug_segfault
                                                std::cerr << "s_rank = " << s_rank << ";\t" << "z_i = " << z_i << ";\t" << "s_rank - z_i = " << as_integers(step_tmp)[1] << std::endl; // need to be different with previous z_i? Could be the same. 
#endif
                                            }

                                            // z_i = z(gen);
                                            // as_integers(step_d)[0] = as_integer(path);
                                            // as_integers(step_d)[1] = s_rank - z_i;
#endif                                       

                                        } else {
                                            // go forward
                                            uint64_t jump_space = std::min(space, path_step_count - s_rank - 1);
                                            uint64_t space = jump_space;
                                            if (jump_space > space_max){
                                                space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                                            }
                                            dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, theta, zetas[space]);
                                            dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                                            uint64_t z_i = z(gen);
                                            //assert(z_i <= path_space);
                                            as_integers(step_b)[0] = as_integer(path);
                                            as_integers(step_b)[1] = s_rank + z_i;

#ifdef increase_compute
                                            steps_in_path.emplace_back(step_b);
                                            for (uint64_t i = 2; i < num_nodes_per_step; ++i) {
                                                // Choose more nodes from the same path in each "step"
                                                z_i = z(gen);
                                                
                                                as_integers(step_tmp)[0] = as_integer(path);
                                                as_integers(step_tmp)[1] = s_rank + z_i;
                                                steps_in_path.emplace_back(step_tmp);
#ifdef debug_segfault
                                                std::cerr << "s_rank = " << s_rank << ";\t" << "z_i = " << z_i << ";\t" << "s_rank - z_i = " << as_integers(step_tmp)[1] << std::endl; // need to be different with previous z_i? Could be the same. 
#endif                                                
                                            }

                                            // // Choose more nodes from the same path in each "step"
                                            // z_i = z(gen);
                                            // // std::cerr << "another z_i: " << z_i << std::endl; // need to be different with previous z_i? Could be the same. 
                                            // as_integers(step_c)[0] = as_integer(path);
                                            // as_integers(step_c)[1] = s_rank + z_i;
                                            
                                            // z_i = z(gen);
                                            // as_integers(step_d)[0] = as_integer(path);
                                            // as_integers(step_d)[1] = s_rank + z_i;
#endif
   
                                        }
                                    } else {
                                        // sample randomly across the path
                                        std::uniform_int_distribution<uint64_t> rando(0, graph.get_step_count(path)-1);
                                        as_integers(step_b)[0] = as_integer(path);
                                        as_integers(step_b)[1] = rando(gen);
#ifdef increase_compute

                                        steps_in_path.emplace_back(step_b);
                                        for (uint64_t i = 2; i < num_nodes_per_step; ++i) {
                                            // Choose more nodes from the same path in each "step"
                                            as_integers(step_tmp)[0] = as_integer(path);
                                            as_integers(step_tmp)[1] = rando(gen);
                                            steps_in_path.emplace_back(step_tmp);
#ifdef debug_segfault
                                            std::cerr << "as_integers(step_tmp)[1] = " << as_integers(step_tmp)[1] << std::endl; // need to be different with previous z_i? Could be the same. 
#endif                                            
                                        }
#endif
                                    }
#ifdef debug_segfault
                                    std::cerr << "size of steps_in_path: " << steps_in_path.size() << std::endl;
#endif
                                    // vector to store pos_in_path, coord_idx
                                    std::vector<std::tuple<size_t, uint64_t>> pos_in_path_coord_idx;
                                    pos_in_path_coord_idx.reserve(num_nodes_per_step);
#ifdef debug_segfault
                                    std::cerr << "size of pos_in_path_coord_idx: " << pos_in_path_coord_idx.size() << std::endl;
#endif
                                    // pos_in_path_coord_idx.emplace_back(get_pos_in_path_and_coord_idx(graph,
                                    //                                                                   path_index, 
                                    //                                                                   flip,
                                    //                                                                   gen,
                                    //                                                                   step_a)
                                    //                                   );

                                    // pos_in_path_coord_idx.emplace_back(get_pos_in_path_and_coord_idx(graph,
                                    //                                                                   path_index, 
                                    //                                                                   flip,
                                    //                                                                   gen,
                                    //                                                                   step_b)
                                    //                                   );

                                    // auto [pos_in_path_a, coord_idx_a] = get_pos_in_path_and_coord_idx(graph,
                                    //                                                                   path_index, 
                                    //                                                                   flip,
                                    //                                                                   gen,
                                    //                                                                   step_a);

                                    // auto [pos_in_path_b, coord_idx_b] = get_pos_in_path_and_coord_idx(graph,
                                    //                                                                   path_index, 
                                    //                                                                   flip,
                                    //                                                                   gen,
                                    //                                                                   step_b);

#ifdef increase_compute
                                    for (auto& step : steps_in_path) {
#ifdef debug_segfault
                                        std::cerr << "as_integers(step)[1]: " << as_integers(step)[1] << std::endl;
#endif
                                        pos_in_path_coord_idx.emplace_back(get_pos_in_path_and_coord_idx(graph,
                                                                                                          path_index, 
                                                                                                          flip,
                                                                                                          gen,
                                                                                                          step)
                                                                          );
#ifdef debug_segfault
                                        std::cerr << "size of pos_in_path_coord_idx: " << pos_in_path_coord_idx.size() << std::endl;
#endif
                                    }
#ifdef debug_segfault
                                    std::cerr << "size of pos_in_path_coord_idx: " << pos_in_path_coord_idx.size() << std::endl;
#endif
                                    // auto [pos_in_path_c, coord_idx_c] = get_pos_in_path_and_coord_idx(graph,
                                    //                                                                   path_index, 
                                    //                                                                   flip,
                                    //                                                                   gen,
                                    //                                                                   step_c);

                                    // auto [pos_in_path_d, coord_idx_d] = get_pos_in_path_and_coord_idx(graph,
                                    //                                                                   path_index, 
                                    //                                                                   flip,
                                    //                                                                   gen,
                                    //                                                                   step_d);
#endif

                                    // // and the graph handles, which we need to record the update
                                    // handle_t term_a = path_index.get_handle_of_step(step_a);
                                    // handle_t term_b = path_index.get_handle_of_step(step_b);
                                    // uint64_t term_a_length = graph.get_length(term_a);
                                    // uint64_t term_b_length = graph.get_length(term_b);

                                    // // adjust the positions to the node starts
                                    // size_t pos_in_path_a = path_index.get_position_of_step(step_a);
                                    // size_t pos_in_path_b = path_index.get_position_of_step(step_b);

                                    // // determine which end we're working with for each node
                                    // bool term_a_is_rev = graph.get_is_reverse(term_a);
                                    // bool use_other_end_a = flip(gen); // 1 == +; 0 == -
                                    // if (use_other_end_a) {
                                    //     pos_in_path_a += term_a_length;
                                    //     // flip back if we were already reversed
                                    //     use_other_end_a = !term_a_is_rev;
                                    // } else {
                                    //     use_other_end_a = term_a_is_rev;
                                    // }
                                    // bool term_b_is_rev = graph.get_is_reverse(term_b);
                                    // bool use_other_end_b = flip(gen); // 1 == +; 0 == -
                                    // if (use_other_end_b) {
                                    //     pos_in_path_b += term_b_length;
                                    //     // flip back if we were already reversed
                                    //     use_other_end_b = !term_b_is_rev;
                                    // } else {
                                    //     use_other_end_b = term_b_is_rev;
                                    // }

#ifdef increase_compute
                                    // step_c, step_d
                                    // handle_t term_c = path_index.get_handle_of_step(step_c);
                                    // handle_t term_d = path_index.get_handle_of_step(step_d);
                                    // uint64_t term_c_length = graph.get_length(term_c);
                                    // uint64_t term_d_length = graph.get_length(term_d);

                                    // // adjust the positions to the node starts
                                    // size_t pos_in_path_c = path_index.get_position_of_step(step_c);
                                    // size_t pos_in_path_d = path_index.get_position_of_step(step_d);

                                    // // determine which end we're working with for each node
                                    // bool term_c_is_rev = graph.get_is_reverse(term_c);
                                    // bool use_other_end_c = flip(gen); // 1 == +; 0 == -
                                    // if (use_other_end_c) {
                                    //     pos_in_path_c += term_c_length;
                                    //     // flip back if we were already reversed
                                    //     use_other_end_c = !term_c_is_rev;
                                    // } else {
                                    //     use_other_end_c = term_c_is_rev;
                                    // }
                                    // bool term_d_is_rev = graph.get_is_reverse(term_d);
                                    // bool use_other_end_d = flip(gen); // 1 == +; 0 == -
                                    // if (use_other_end_d) {
                                    //     pos_in_path_d += term_d_length;
                                    //     // flip back if we were already reversed
                                    //     use_other_end_d = !term_d_is_rev;
                                    // } else {
                                    //     use_other_end_d = term_d_is_rev;
                                    // }                                    

#endif

#ifdef debug_path_sgd
                                    std::cerr << "1. pos in path " << pos_in_path_a << " " << pos_in_path_b << std::endl;
#endif
                                    // assert(pos_in_path_a < path_index.get_path_length(path));
                                    // assert(pos_in_path_b < path_index.get_path_length(path));
#ifdef debug_path_sgd
                                    std::cerr << "2. pos in path " << pos_in_path_a << " " << pos_in_path_b << std::endl;
#endif

#ifdef debug_test
                                    for (auto it = pos_in_path_coord_idx.begin(); it != pos_in_path_coord_idx.end(); ++it) {
                                        std::cerr << "pos in path: " << std::get<0>(*it) << " coord idx: " << std::get<1>(*it) << std::endl;
                                    }
#endif

                                    if (all_node_combinations) { // True for all node combinations
                                        for (auto it_a = pos_in_path_coord_idx.begin(); it_a != pos_in_path_coord_idx.end() - 1; ++it_a) {
                                            for (auto it_b = it_a + 1; it_b != pos_in_path_coord_idx.end(); ++it_b) {
                                                auto [pos_in_path_a, coord_idx_a] = *it_a;
                                                auto [pos_in_path_b, coord_idx_b] = *it_b;
                                                update_pos(pos_in_path_a, pos_in_path_b, coord_idx_a, coord_idx_b, X, Y, eta, Delta_max);
                                            }
                                        }
                                    } else { // false for all node combinations. Still, update N times for 2*N nodes. 
                                        for (auto it = pos_in_path_coord_idx.begin(); it != pos_in_path_coord_idx.end(); it = it + 2) {
                                            auto [pos_in_path_a, coord_idx_a] = *it;
                                            auto [pos_in_path_b, coord_idx_b] = *(it + 1);
                                            update_pos(pos_in_path_a, pos_in_path_b, coord_idx_a, coord_idx_b, X, Y, eta, Delta_max);
                                        }
                                    }



#ifdef debug_segfault                                        
                                        std::cerr << "pos in path 0: " << pos_in_path_0 <<"\t" << "pos in path 1: " << pos_in_path_1 << std::endl;
#endif



                                    // update_pos(pos_in_path_a, pos_in_path_b, coord_idx_a, coord_idx_b, X, Y, eta, Delta_max);

#ifdef increase_compute
                                    // update_pos(pos_in_path_c, pos_in_path_d, coord_idx_c, coord_idx_d, X, Y, eta, Delta_max);
#endif

#ifdef more_pairs
                                    // a, c
                                    update_pos(pos_in_path_a, pos_in_path_c, coord_idx_a, coord_idx_c, X, Y, eta, Delta_max);
                                    // b, d
                                    update_pos(pos_in_path_b, pos_in_path_d, coord_idx_b, coord_idx_d, X, Y, eta, Delta_max);
                                    // a, d
                                    update_pos(pos_in_path_a, pos_in_path_d, coord_idx_a, coord_idx_d, X, Y, eta, Delta_max);
                                    // b, c
                                    update_pos(pos_in_path_b, pos_in_path_c, coord_idx_b, coord_idx_c, X, Y, eta, Delta_max);
#endif

/*
                                    // establish the term distance
                                    double term_dist = std::abs(
                                            static_cast<double>(pos_in_path_a) - static_cast<double>(pos_in_path_b));

                                    if (term_dist == 0) {
                                        term_dist = 1e-9;
                                    }
                                    
#ifdef increase_compute
                                    // TODO: cross-compute over 4 nodes -> we got 6 node pairs. 
#endif


#ifdef eval_path_sgd
                                    std::string path_name = path_index.get_path_name(path);
                                std::cerr << path_name << "\t" << pos_in_path_a << "\t" << pos_in_path_b << "\t" << term_dist << std::endl;
#endif
                                    // assert(term_dist == zipf_int);
#ifdef debug_path_sgd
                                    std::cerr << "term_dist: " << term_dist << std::endl;
#endif
                                    double term_weight = 1.0 / (double) term_dist;

                                    double w_ij = term_weight;
#ifdef debug_path_sgd
                                    std::cerr << "w_ij = " << w_ij << std::endl;
#endif
                                    double mu = eta.load() * w_ij;
                                    if (mu > 1) {
                                        mu = 1;
                                    }
                                    // actual distance in graph
                                    double d_ij = term_dist;
                                    // identities
                                    // uint64_t i = number_bool_packing::unpack_number(term_i);
                                    // uint64_t j = number_bool_packing::unpack_number(term_j);
#ifdef increase_compute
                                    // TODO: cross-compute over 4 nodes -> we got 6 node pairs. 
#endif

#ifdef debug_path_sgd
                                    #pragma omp critical (cerr)
                                std::cerr << "nodes are " << graph.get_id(term_i) << " and " << graph.get_id(term_j) << std::endl;
#endif
                                    // distance == magnitude in our 2D situation
                                    // uint64_t offset_i = 0;
                                    // uint64_t offset_j = 0;
                                    // if (use_other_end_a) {
                                    //     offset_i += 1;
                                    // }
                                    // if (use_other_end_b) {
                                    //     offset_j += 1;
                                    // }
                                    // double dx = X[2 * i + offset_i].load() - X[2 * j + offset_j].load();
                                    // double dy = Y[2 * i + offset_i].load() - Y[2 * j + offset_j].load();
                                    
                                    double dx = X[coord_idx_a].load() - X[coord_idx_b].load();
                                    double dy = Y[coord_idx_a].load() - Y[coord_idx_b].load();                                    
                                    if (dx == 0) {
                                        dx = 1e-9; // avoid nan
                                    }
#ifdef debug_path_sgd
                                    #pragma omp critical (cerr)
                                std::cerr << "distance is " << dx << " but should be " << d_ij << std::endl;
#endif
                                    //double mag = dx; //sqrt(dx*dx + dy*dy);
                                    double mag = sqrt(dx * dx + dy * dy);
#ifdef debug_path_sgd
                                    std::cerr << "mu " << mu << " mag " << mag << " d_ij " << d_ij << std::endl;
#endif
                                    // check distances for early stopping
                                    double Delta = mu * (mag - d_ij) / 2;
                                    // try until we succeed. risky.
                                    double Delta_abs = std::abs(Delta);
#ifdef debug_path_sgd
                                    #pragma omp critical (cerr)
                                std::cerr << "Delta_abs " << Delta_abs << std::endl;
#endif
                                    // todo use atomic compare and swap
                                    while (Delta_abs > Delta_max.load()) {
                                        Delta_max.store(Delta_abs);
                                    }
                                    // calculate update
                                    double r = Delta / mag;
                                    double r_x = r * dx;
                                    double r_y = r * dy;
#ifdef debug_path_sgd
                                    #pragma omp critical (cerr)
                                std::cerr << "r_x is " << r_x << std::endl;
#endif
                                    // update our positions (atomically)
#ifdef debug_path_sgd
                                    std::cerr << "before X[i] " << X[i].load() << " X[j] " << X[j].load() << std::endl;
#endif
                                    // X[2 * i + offset_i].store(X[2 * i + offset_i].load() - r_x);
                                    // Y[2 * i + offset_i].store(Y[2 * i + offset_i].load() - r_y);
                                    // X[2 * j + offset_j].store(X[2 * j + offset_j].load() + r_x);
                                    // Y[2 * j + offset_j].store(Y[2 * j + offset_j].load() + r_y);
                                    X[coord_idx_a].store(X[coord_idx_a].load() - r_x);
                                    Y[coord_idx_a].store(Y[coord_idx_a].load() - r_y);
                                    X[coord_idx_b].store(X[coord_idx_b].load() + r_x);
                                    Y[coord_idx_b].store(Y[coord_idx_b].load() + r_y);
*/

#ifdef debug_path_sgd
                                    std::cerr << "after X[i] " << X[i].load() << " X[j] " << X[j].load() << std::endl;
#endif
                                    steps_since_term_updates++;

#ifdef debug_test
                                    if (steps_since_term_updates > 10) {
                                        quick_exit(1);
                                    }
#endif
                                    if ((steps_since_term_updates % 1000) == 0) {
                                        term_updates += steps_since_term_updates;
                                        steps_since_term_updates = 0;
                                    }
                                    if (progress) {
                                        progress_meter->increment(1);
                                    }
                                }
                            }
                        };

                auto snapshot_lambda =
                        [&]() {
                            uint64_t iter = 0;
                            while (snapshot && work_todo.load()) {
                                if ((iter < iteration) && iteration != iter_max) {
                                    std::cerr << "[odgi::path_linear_sgd_layout] snapshot thread: Taking snapshot!" << std::endl;
                                    // drop out of atomic stuff... maybe not the best way to do this
                                    std::vector<double> X_iter(X.size());
                                    uint64_t i = 0;
                                    for (auto &x : X) {
                                        X_iter[i++] = x.load();
                                    }
                                    std::vector<double> Y_iter(Y.size());
                                    i = 0;
                                    for (auto &y : Y) {
                                        Y_iter[i++] = y.load();
                                    }
                                    algorithms::layout::Layout layout(X_iter, Y_iter);
                                    std::string local_snapshot_prefix = snapshot_prefix + std::to_string(iter + 1);
                                    ofstream snapshot_out(local_snapshot_prefix);
                                    // write out
                                    layout.serialize(snapshot_out);
                                    iter = iteration;
                                    snapshot_in_progress.store(false);
                                    snapshot_progress[iter].store(true);
                                }
                                std::this_thread::sleep_for(1ms);
                            }

                        };

                std::thread checker(checker_lambda);
                std::thread snapshot_thread(snapshot_lambda);

                std::vector<std::thread> workers;
                workers.reserve(nthreads);
                for (uint64_t t = 0; t < nthreads; ++t) {
                    workers.emplace_back(worker_lambda, t);
                }

                for (uint64_t t = 0; t < nthreads; ++t) {
                    workers[t].join();
                }

                snapshot_thread.join();

                checker.join();
            }

            if (progress) {
                progress_meter->finish();
            }
        }

        std::vector<double> path_linear_sgd_layout_schedule(const double &w_min,
                                                            const double &w_max,
                                                            const uint64_t &iter_max,
                                                            const uint64_t &iter_with_max_learning_rate,
                                                            const double &eps) {
#ifdef debug_schedule
            std::cerr << "w_min: " << w_min << std::endl;
            std::cerr << "w_max: " << w_max << std::endl;
            std::cerr << "iter_max: " << iter_max << std::endl;
            std::cerr << "eps: " << eps << std::endl;
#endif
            double eta_max = 1.0 / w_min;
            double eta_min = eps / w_max;
            double lambda = log(eta_max / eta_min) / ((double) iter_max - 1);
#ifdef debug_schedule
            std::cerr << "eta_max: " << eta_max << std::endl;
            std::cerr << "eta_min: " << eta_min << std::endl;
            std::cerr << "lambda: " << lambda << std::endl;
#endif
            // initialize step sizes
            std::vector<double> etas;
            etas.reserve(iter_max+1);
#ifdef debug_schedule
            std::cerr << "etas: ";
#endif
            for (int64_t t = 0; t <= iter_max; t++) {
                etas.push_back(eta_max * exp(-lambda * (abs(t - (int64_t) iter_with_max_learning_rate))));
#ifdef debug_schedule
                std::cerr << etas.back() << ", ";
#endif
            }
#ifdef debug_schedule
            std::cerr << std::endl;
#endif
            return etas;
        }

        // function to get "pos_in_path" and "coord_idx" for a given step
        // return tuple: (size_t pos_in_path_a, uint64_t coord_index)
        std::tuple<size_t, uint64_t> get_pos_in_path_and_coord_idx(const PathHandleGraph &graph,
                                                                   const xp::XP &path_index, 
                                                                   std::uniform_int_distribution<uint64_t> &flip,
                                                                   XoshiroCpp::Xoshiro256Plus &gen,
                                                                   step_handle_t &step) {

            // and the graph handles, which we need to record the update
            handle_t term = path_index.get_handle_of_step(step);
            uint64_t term_length = graph.get_length(term);

            // adjust the positions to the node starts
            size_t pos_in_path = path_index.get_position_of_step(step);

            // determine which end we're working with for each node
            bool term_is_rev = graph.get_is_reverse(term);
            bool use_other_end = flip(gen); // 1 == +; 0 == -
            if (use_other_end) {
                pos_in_path += term_length;
                // flip back if we were already reversed
                use_other_end = !term_is_rev;
            } else {
                use_other_end = term_is_rev;
            }
            
            uint64_t i = number_bool_packing::unpack_number(term);

            uint64_t offset = 0;
            if (use_other_end) {
                offset += 1;
            }

            uint64_t idx = 2 * i + offset;
            return std::make_tuple(pos_in_path, idx);
        }

        // function to update X, Y
        void update_pos(size_t &pos_in_path_a, 
                        size_t &pos_in_path_b,
                        uint64_t &coord_idx_a,
                        uint64_t &coord_idx_b,
                        std::vector<std::atomic<double>> &X,
                        std::vector<std::atomic<double>> &Y, 
                        std::atomic<double> &eta, 
                        std::atomic<double> &Delta_max) {
                            // establish the term distance
                            double term_dist = std::abs(static_cast<double>(pos_in_path_a) - static_cast<double>(pos_in_path_b));

                            // std::string path_name = path_index.get_path_name(path);


                            if (term_dist == 0) {
                                term_dist = 1e-9;
                            }
#ifdef debug_test
                            std::cerr << pos_in_path_a << "\t" << pos_in_path_b << "\t" << term_dist << std::endl;
#endif
                            double term_weight = 1.0 / (double) term_dist;
                            double mu = eta.load() * term_weight;
                            if (mu > 1) {
                                mu = 1;
                            }

                            double dx = X[coord_idx_a].load() - X[coord_idx_b].load();
                            double dy = Y[coord_idx_a].load() - Y[coord_idx_b].load();                         

                            if (dx == 0) {
                                dx = 1e-9; // avoid nan
                            }

                            double mag = sqrt(dx * dx + dy * dy);

                            // check distances for early stopping
                            double Delta = mu * (mag - term_dist) / 2;
                            // try until we succeed. risky.
                            double Delta_abs = std::abs(Delta);

// #ifdef debug_test
//                             std::cerr << "Delta_abs: " << Delta_abs << std::endl;
// #endif
                            // todo use atomic compare and swap
                            while (Delta_abs > Delta_max.load()) {
// #ifdef debug_test
//                                 std::cerr << "before: Delta_max: " << Delta_max.load() << std::endl;
// #endif
                                Delta_max.store(Delta_abs);
// #ifdef debug_test
//                                 std::cerr << "after: Delta_max: " << Delta_max.load() << std::endl;
// #endif
                            }
                            // calculate update
                            double r = Delta / mag;
                            double r_x = r * dx;
                            double r_y = r * dy;

                            X[coord_idx_a].store(X[coord_idx_a].load() - r_x);
                            Y[coord_idx_a].store(Y[coord_idx_a].load() - r_y);
                            X[coord_idx_b].store(X[coord_idx_b].load() + r_x);
                            Y[coord_idx_b].store(Y[coord_idx_b].load() + r_y);
                            return;
                        }
    }
}
