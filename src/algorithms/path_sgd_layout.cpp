#include "path_sgd_layout.hpp"
#include "algorithms/layout.hpp"

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
                                    std::vector<std::atomic<double>> &Y) {
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
            cuda_hello_host();

            tf::Executor executor;
            tf::Taskflow taskflow;

            uint64_t first_cooling_iteration = std::floor(cooling_start * (double)iter_max);
            //std::cerr << "first cooling iteration " << first_cooling_iteration << std::endl;

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

                // learning rate
                std::atomic<double> eta;
                eta.store(etas.front());
                // if we're in a final cooling phase (last 10%) of iterations
                std::atomic<bool> cooling;
                // current iteration
                volatile uint64_t iteration = 0;


                auto init_lambda =
                    [&]() {
                        iteration = 1;
                    };


                // launch a thread to update the learning rate, count iterations, and decide when to stop
                auto config_params_lambda =
                        [&]() {
                            eta.store(etas[iteration-1]); // update our learning rate
                            if (iteration >= first_cooling_iteration) {
                                cooling.store(true);
                            } else {
                                cooling.store(false);
                            }
                            std::cout << "starting iteration: " << iteration << " step size: " << eta.load() << " cooling: " << cooling.load() << std::endl;
                        };


                // some references to literal bitvectors in the path index hmmm
                const sdsl::bit_vector &np_bv = path_index.get_np_bv();
                const sdsl::int_vector<> &nr_iv = path_index.get_nr_iv();
                const sdsl::int_vector<> &npi_iv = path_index.get_npi_iv();

                atomic<uint64_t> seed = 939920;
                // we'll sample from all path steps
                std::uniform_int_distribution<uint64_t> dis_step = std::uniform_int_distribution<uint64_t>(0, np_bv.size() - 1);
                std::uniform_int_distribution<uint64_t> flip(0, 1);

                uint64_t nbr_tasks = 10000;
                uint64_t nbr_loops = (min_term_updates / nbr_tasks) + 1;
                if (min_term_updates < nbr_tasks) {
                    nbr_tasks = min_term_updates;
                    nbr_loops = 1;
                }

                auto step_lambda =
                        [&]() {
                            // everyone tries to seed with their own random data
                            XoshiroCpp::Xoshiro256Plus gen{seed++}; // a nice, fast PRNG

                            for (int loop = 0; loop < nbr_loops; loop++) {
                                // sample the first node from all the nodes in the graph
                                // pick a random position from all paths
                                uint64_t step_index = dis_step(gen);
#ifdef debug_sample_from_nodes
                                std::cerr << "step_index: " << step_index << std::endl;
#endif
                                uint64_t path_i = npi_iv[step_index];
                                path_handle_t path = as_path_handle(path_i);
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
                                size_t path_step_count = path_index.get_path_step_count(path);
                                // if (path_step_count == 1){
                                //     continue;
                                // }

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
                                    }
                                } else {
                                    // sample randomly across the path
                                    graph.get_step_count(path);
                                    std::uniform_int_distribution<uint64_t> rando(0, graph.get_step_count(path)-1);
                                    as_integers(step_b)[0] = as_integer(path);
                                    as_integers(step_b)[1] = rando(gen);
                                }


                                // and the graph handles, which we need to record the update
                                handle_t term_i = path_index.get_handle_of_step(step_a);
                                handle_t term_j = path_index.get_handle_of_step(step_b);
                                uint64_t term_i_length = graph.get_length(term_i);
                                uint64_t term_j_length = graph.get_length(term_j);

                                // adjust the positions to the node starts
                                size_t pos_in_path_a = path_index.get_position_of_step(step_a);
                                size_t pos_in_path_b = path_index.get_position_of_step(step_b);

                                // determine which end we're working with for each node
                                bool term_i_is_rev = graph.get_is_reverse(term_i);
                                bool use_other_end_a = flip(gen); // 1 == +; 0 == -
                                if (use_other_end_a) {
                                    pos_in_path_a += term_i_length;
                                    // flip back if we were already reversed
                                    use_other_end_a = !term_i_is_rev;
                                } else {
                                    use_other_end_a = term_i_is_rev;
                                }
                                bool term_j_is_rev = graph.get_is_reverse(term_j);
                                bool use_other_end_b = flip(gen); // 1 == +; 0 == -
                                if (use_other_end_b) {
                                    pos_in_path_b += term_j_length;
                                    // flip back if we were already reversed
                                    use_other_end_b = !term_j_is_rev;
                                } else {
                                    use_other_end_b = term_j_is_rev;
                                }

#ifdef debug_path_sgd
                                std::cerr << "1. pos in path " << pos_in_path_a << " " << pos_in_path_b << std::endl;
#endif
                                // assert(pos_in_path_a < path_index.get_path_length(path));
                                // assert(pos_in_path_b < path_index.get_path_length(path));
#ifdef debug_path_sgd
                                std::cerr << "2. pos in path " << pos_in_path_a << " " << pos_in_path_b << std::endl;
#endif
                                // establish the term distance
                                double term_dist = std::abs(
                                        static_cast<double>(pos_in_path_a) - static_cast<double>(pos_in_path_b));

                                if (term_dist == 0) {
                                    term_dist = 1e-9;
                                }
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
                                uint64_t i = number_bool_packing::unpack_number(term_i);
                                uint64_t j = number_bool_packing::unpack_number(term_j);
#ifdef debug_path_sgd
#pragma omp critical (cerr)
                                std::cerr << "nodes are " << graph.get_id(term_i) << " and " << graph.get_id(term_j) << std::endl;
#endif
                                // distance == magnitude in our 2D situation
                                uint64_t offset_i = 0;
                                uint64_t offset_j = 0;
                                if (use_other_end_a) {
                                    offset_i += 1;
                                }
                                if (use_other_end_b) {
                                    offset_j += 1;
                                }
                                double dx = X[2 * i + offset_i].load() - X[2 * j + offset_j].load();
                                double dy = Y[2 * i + offset_i].load() - Y[2 * j + offset_j].load();
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
                                X[2 * i + offset_i].store(X[2 * i + offset_i].load() - r_x);
                                Y[2 * i + offset_i].store(Y[2 * i + offset_i].load() - r_y);
                                X[2 * j + offset_j].store(X[2 * j + offset_j].load() + r_x);
                                Y[2 * j + offset_j].store(Y[2 * j + offset_j].load() + r_y);
#ifdef debug_path_sgd
                                std::cerr << "after X[i] " << X[i].load() << " X[j] " << X[j].load() << std::endl;
#endif
                            }
                        };

                auto iter_incr_lambda =
                    [&]() {
                        iteration++;
                        return (iteration <= iter_max) ? false : true;
                    };


                tf::Task task_init = taskflow.emplace(init_lambda).name("initialization");
                tf::Task task_config_params = taskflow.emplace(config_params_lambda).name("configure_parameters");
                std::vector<tf::Task> step{nbr_tasks};
                for (int i = 0; i < nbr_tasks; i++) {
                    step[i] = taskflow.emplace(step_lambda).name("step_" + to_string(i+1));
                }
                tf::Task task_incr_iter = taskflow.emplace(iter_incr_lambda).name("increment_iteration");
                tf::Task task_finished = taskflow.emplace([](){}).name("finished");

                task_config_params.succeed(task_init);
                for (int i = 0; i < nbr_tasks; i++) {
                    step[i].succeed(task_config_params);
                    task_incr_iter.succeed(step[i]);
                }
                task_incr_iter.precede(task_config_params, task_finished);

                taskflow.dump(std::cout);

                executor.run(taskflow).wait();
            }

            // drop out of atomic stuff... maybe not the best way to do this
            std::vector<double> X_final(X.size());
            uint64_t i = 0;
            for (auto &x : X) {
                X_final[i++] = x.load();
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
    }
}
