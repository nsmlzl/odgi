#include "sort.h"
//#include <cuda.h>


namespace extended_sort {


void cpu_sort(extended_sort::layout_config_t config, double *etas, double *zetas, extended_sort::node_data_t &node_data, extended_sort::path_data_t &path_data) {
    int nbr_threads = config.nthreads;
    std::cout << "cuda cpu sort (" << nbr_threads << " threads)" << std::endl;
    std::vector<uint64_t> path_dist;
    for (int p = 0; p < path_data.path_count; p++) {
        path_dist.push_back(uint64_t(path_data.paths[p].step_count));
    }

#pragma omp parallel num_threads(nbr_threads)
    {
        int tid = omp_get_thread_num();

        XoshiroCpp::Xoshiro256Plus gen(9399220 + tid);
        std::uniform_int_distribution<uint64_t> flip(0, 1);
        std::discrete_distribution<> rand_path(path_dist.begin(), path_dist.end());

        const int steps_per_thread = config.min_term_updates / nbr_threads;

//#define profiling
#ifdef profiling
        auto total_duration_dist = std::chrono::duration<double>::zero(); // total time on computing distance: in seconds
        auto total_duration_sgd = std::chrono::duration<double>::zero(); // total time on SGD: in seconds
        // detailed analysis on different parts of Updating Coordinates Part
        auto total_duration_compute_first = std::chrono::duration<double>::zero();
        auto total_duration_load = std::chrono::duration<double>::zero();
        auto total_duration_compute_second = std::chrono::duration<double>::zero();
        auto total_duration_store = std::chrono::duration<double>::zero();
        // detailed analysis on different parts of Getting Distance Part
        auto total_duration_one_step_gen = std::chrono::duration<double>::zero();
        auto total_duration_two_step_gen = std::chrono::duration<double>::zero();
        auto total_duration_get_distance = std::chrono::duration<double>::zero();


        std::chrono::high_resolution_clock::time_point start_dist;
        std::chrono::high_resolution_clock::time_point end_dist;
        std::chrono::high_resolution_clock::time_point start_sgd;
        std::chrono::high_resolution_clock::time_point one_step_gen;
        std::chrono::high_resolution_clock::time_point two_step_gen;

        // detailed analysis on Updating Coordinates part
        std::chrono::high_resolution_clock::time_point before_load;
        std::chrono::high_resolution_clock::time_point after_load;
        std::chrono::high_resolution_clock::time_point before_store;
        std::chrono::high_resolution_clock::time_point after_store;
#endif

        for (int iter = 0; iter < config.iter_max; iter++ ) {
            // synchronize all threads before each iteration
#pragma omp barrier
            for (int step = 0; step < steps_per_thread; step++ ) {
#ifdef profiling
                start_dist = std::chrono::high_resolution_clock::now();
#endif
                // get path
                uint32_t path_idx = rand_path(gen);
                path_t p = path_data.paths[path_idx];
                if (p.step_count < 2) {
                    continue;
                }

                std::uniform_int_distribution<uint32_t> rand_step(0, p.step_count-1);

                uint32_t s1_idx = rand_step(gen);
#ifdef profiling
                one_step_gen = std::chrono::high_resolution_clock::now();
                total_duration_one_step_gen += std::chrono::duration_cast<std::chrono::nanoseconds>(one_step_gen - start_dist);
#endif
                uint32_t s2_idx;
                //bool cooling = (iter + 1 >= config.first_cooling_iteration)? true: false;
                bool cooling = (iter > config.first_cooling_iteration)? true: false;
                if (cooling || flip(gen)) {
                    double _theta = (cooling)? 0.001: config.theta;
                    if (s1_idx > 0 && flip(gen) || s1_idx == p.step_count-1) {
                        // go backward
                        uint64_t jump_space = std::min(config.space, (uint64_t) s1_idx);
                        uint64_t space = jump_space;
                        if (jump_space > config.space_max) {
                            space = config.space_max + (jump_space - config.space_max) / config.space_quantization_step + 1;
                        }
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, _theta, zetas[space]);
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                        uint32_t z_i = (uint32_t) z(gen);
                        s2_idx = s1_idx - z_i;
                    } else {
                        // go forward
                        uint64_t jump_space = std::min(config.space, (uint64_t) (p.step_count - s1_idx - 1));
                        uint64_t space = jump_space;
                        if (jump_space > config.space_max) {
                            space = config.space_max + (jump_space - config.space_max) / config.space_quantization_step + 1;
                        }
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, _theta, zetas[space]);
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                        uint32_t z_i = (uint32_t) z(gen);
                        s2_idx = s1_idx + z_i;
                    }
                } else {
                    do {
                        s2_idx = rand_step(gen);
                    } while (s1_idx == s2_idx);
                }
#ifdef profiling
                two_step_gen = std::chrono::high_resolution_clock::now();
                total_duration_two_step_gen += std::chrono::duration_cast<std::chrono::nanoseconds>(two_step_gen - one_step_gen);
#endif
                // TODO add target sorting

                uint32_t n1_id = p.elements[s1_idx].node_id;
                int64_t n1_pos_in_path = p.elements[s1_idx].pos;
                bool n1_is_rev = (n1_pos_in_path < 0)? true: false;
                n1_pos_in_path = std::abs(n1_pos_in_path);

                uint32_t n2_id = p.elements[s2_idx].node_id;
                int64_t n2_pos_in_path = p.elements[s2_idx].pos;
                bool n2_is_rev = (n2_pos_in_path < 0)? true: false;
                n2_pos_in_path = std::abs(n2_pos_in_path);

                double term_dist = std::abs(static_cast<double>(n1_pos_in_path) - static_cast<double>(n2_pos_in_path));

                if (term_dist == 0.0) {
                    continue;
                }
#ifdef profiling
                end_dist = std::chrono::high_resolution_clock::now();
                total_duration_get_distance += std::chrono::duration_cast<std::chrono::nanoseconds>(end_dist - two_step_gen);

                total_duration_dist += std::chrono::duration_cast<std::chrono::nanoseconds>(end_dist - start_dist);

                start_sgd = std::chrono::high_resolution_clock::now();
#endif

                double w_ij = 1.0 / term_dist;

                double mu = etas[iter] * w_ij;
                if (mu > 1.0) {
                    mu = 1.0;
                }

                double d_ij = term_dist;

#ifdef profiling
                before_load = std::chrono::high_resolution_clock::now();
                total_duration_compute_first += std::chrono::duration_cast<std::chrono::nanoseconds>(before_load - start_sgd);
#endif
                double *x1 = &node_data.nodes[n1_id].X;
                double *x2 = &node_data.nodes[n2_id].X;

                double dx = double(*x1 - *x2);
#ifdef profiling
                after_load = std::chrono::high_resolution_clock::now();
                total_duration_load += std::chrono::duration_cast<std::chrono::nanoseconds>(after_load - before_load);
#endif
                if (dx == 0.0) {
                    dx = 1e-9;
                }

                double mag = std::abs(dx);
                double delta = mu * (mag - d_ij) / 2.0;
                //double delta_abs = std::abs(delta);

                // TODO implement delta max stop functionality
                double r = delta / mag;
                double r_x = r * dx;

#ifdef profiling
                before_store = std::chrono::high_resolution_clock::now();
                total_duration_compute_second += std::chrono::duration_cast<std::chrono::nanoseconds>(before_store - after_load);
#endif
                *x1 -= double(r_x);
                *x2 += double(r_x);
#ifdef profiling
                after_store = std::chrono::high_resolution_clock::now();
                total_duration_store += std::chrono::duration_cast<std::chrono::nanoseconds>(after_store - before_store);
                total_duration_sgd += std::chrono::duration_cast<std::chrono::nanoseconds>(after_store - start_sgd);
#endif
            }
        }

#ifdef profiling
        std::stringstream msg;
        msg << "Thread[" << tid << "]: Dataloading time = " << total_duration_dist.count() << " sec;\t" << "Compute time = " << total_duration_sgd.count() << " sec." << std::endl;

        msg << std::left
            << std::setw(40) << "Getting Distance Part Breakdown: " << std::endl
            << std::setw(20) << "[0] One Step Gen: "
            << std::setw(10) << total_duration_one_step_gen.count()
            << std::setw(10)  << " sec;"
            << std::setw(20) << "[1] Two Steps Gen: "
            << std::setw(10) << total_duration_two_step_gen.count()
            << std::setw(10)  << " sec;"
            << std::setw(20) << "[2] Get Distance: "
            << std::setw(10) << total_duration_get_distance.count()
            << std::setw(10) << " sec."
            << std::endl;

        msg << std::setw(40) << "Updating Coordinate Part Breakdown: " << std::endl
            << std::setw(20) << "[0] First Compute: "
            << std::setw(10) << total_duration_compute_first.count()
            << std::setw(10)  << " sec;"
            << std::setw(20) << "[1] Load Pos: "
            << std::setw(10) << total_duration_load.count()
            << std::setw(10)  << " sec;"
            << std::setw(20) << "[2] Second Compute: "
            << std::setw(10) << total_duration_compute_second.count()
            << std::setw(10)  << " sec;"
            << std::setw(20) << "[3] Update Pos: "
            << std::setw(10) << total_duration_store.count()
            << std::setw(10)  << " sec."
            << std::endl << std::endl;

        std::cerr << msg.str();
#endif

    }
}


std::vector<double> cache_optimized_sort(layout_config_t config, const odgi::graph_t &graph, const xp::XP &path_index) {
    std::vector<double> X(graph.get_node_count());

#ifdef cuda_layout_profiling
    auto start = std::chrono::high_resolution_clock::now();
#endif


    std::cout << "Hello world from CUDA host" << std::endl;
    std::cout << "iter_max: " << config.iter_max << std::endl;
    std::cout << "first_cooling_iteration: " << config.first_cooling_iteration << std::endl;
    std::cout << "min_term_updates: " << config.min_term_updates << std::endl;
    std::cout << "size of node_t: " << sizeof(node_t) << std::endl;

    // create eta array
    double *etas;
    //cudaMallocManaged(&etas, config.iter_max * sizeof(double));
    etas = (double*) malloc(config.iter_max * sizeof(double));

    const int32_t iter_max = config.iter_max;
    const int32_t iter_with_max_learning_rate = config.iter_with_max_learning_rate;
    const double w_max = 1.0;
    const double eps = config.eps;
    const double eta_max = config.eta_max;
    const double eta_min = eps / w_max;
    const double lambda = log(eta_max / eta_min) / ((double) iter_max - 1);
    for (int32_t i = 0; i < config.iter_max; i++) {
        double eta = eta_max * exp(-lambda * (std::abs(i - iter_with_max_learning_rate)));
        etas[i] = isnan(eta)? eta_min : eta;
        //std::cout << "eta " << i << " : " << eta << std::endl;
    }


    // create node data structure
    // consisting of sequence length and coords
    uint32_t node_count = graph.get_node_count();
    std::cout << "node_count: " << node_count << std::endl;
    // TODO handle cases when min_node_id != 1
    assert(graph.min_node_id() == 1);
    assert(graph.max_node_id() == node_count);
    assert(graph.max_node_id() - graph.min_node_id() + 1 == node_count);

    extended_sort::node_data_t node_data;
    node_data.node_count = node_count;
    //cudaMallocManaged(&node_data.nodes, node_count * sizeof(cuda::node_t));
    node_data.nodes = (extended_sort::node_t*) malloc(node_count * sizeof(extended_sort::node_t));
    // NOTE Do not parallelise (len_global counter)
    uint64_t len_global = 0;
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        assert(graph.has_node(node_idx + 1));
        extended_sort::node_t *n_tmp = &node_data.nodes[node_idx];

        // sequence length
        const handlegraph::handle_t h = graph.get_handle(node_idx + 1, false);
        // NOTE: unable store orientation (reverse), since this information is path dependent
        //n_tmp->seq_length = graph.get_length(h);

        // init coordinate
        n_tmp->X = double(len_global);
        len_global += graph.get_length(h);
    }


    // create path data structure
    uint32_t path_count = graph.get_path_count();
    extended_sort::path_data_t path_data;
    path_data.path_count = path_count;
    path_data.total_path_steps = 0;
    //cudaMallocManaged(&path_data.paths, node_count * sizeof(cuda::path_t));
    path_data.paths = (extended_sort::path_t*) malloc(node_count * sizeof(extended_sort::path_t));

    vector<odgi::path_handle_t> path_handles{};
    path_handles.reserve(path_count);
    graph.for_each_path_handle(
        [&] (const odgi::path_handle_t& p) {
            path_handles.push_back(p);
            path_data.total_path_steps += graph.get_step_count(p);
        });

#pragma omp parallel for num_threads(config.nthreads)
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        // TODO: sort paths for uniform distribution? Largest should not just be next to each other
        odgi::path_handle_t p = path_handles[path_idx];
        //std::cout << graph.get_path_name(p) << ": " << graph.get_step_count(p) << std::endl;

        int step_count = graph.get_step_count(p);
        path_data.paths[path_idx].step_count = step_count;

        extended_sort::path_element_t *cur_path;
        //cudaMallocManaged(&cur_path, step_count * sizeof(path_element_t));
        cur_path = (extended_sort::path_element_t*) malloc(step_count * sizeof(path_element_t));
        path_data.paths[path_idx].elements = cur_path;

        odgi::step_handle_t s = graph.path_begin(p);
        int64_t pos = 1;
        // Iterate through path
        for (int step_idx = 0; step_idx < step_count; step_idx++) {
            // TODO get position with path_index? see line 356
            odgi::handle_t h = graph.get_handle_of_step(s);
            //std::cout << graph.get_id(h) << std::endl;

            cur_path[step_idx].node_id = graph.get_id(h) - 1;
            // store position negative when handle reverse
            if (graph.get_is_reverse(h)) {
                cur_path[step_idx].pos = -pos;
            }  else {
                cur_path[step_idx].pos = pos;
            }
            pos += graph.get_length(h);

            //cur_path[step_idx].pos = int64_t(path_index.get_position_of_step(s));

            // get next step
            if (graph.has_next_step(s)) {
                s = graph.get_next_step(s);
            } else if (!(step_idx == step_count-1)) {
                // should never be reached
                std::cout << "Error: Here should be another step" << std::endl;
            }
        }
    }


    // cache zipf zetas
    double *zetas;
    uint64_t zetas_cnt = ((config.space <= config.space_max)? config.space : (config.space_max + (config.space - config.space_max) / config.space_quantization_step + 1)) + 1;
    //cudaMallocManaged(&zetas, zetas_cnt * sizeof(double));
    zetas = (double*) malloc(zetas_cnt * sizeof(double));
    uint64_t last_quantized_i = 0;
    // TODO parallelise with openmp?
    for (uint64_t i = 1; i < config.space + 1; i++) {
        uint64_t quantized_i = i;
        uint64_t compressed_space = i;
        if (i > config.space_max) {
            quantized_i = config.space_max + (i - config.space_max) / config.space_quantization_step + 1;
            compressed_space = config.space_max + ((i - config.space_max) / config.space_quantization_step) * config.space_quantization_step;
        }

        if (quantized_i != last_quantized_i) {
            dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, compressed_space, config.theta);
            zetas[quantized_i] = z_p.zeta();
            last_quantized_i = quantized_i;
        }
    }



    auto start_compute = std::chrono::high_resolution_clock::now();

    cpu_sort(config, etas, zetas, node_data, path_data);

    auto end_compute = std::chrono::high_resolution_clock::now();
    uint32_t duration_compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    std::cout << "CUDA layout compute took " << duration_compute_ms << "ms" << std::endl;



    // copy coord to X vector
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        extended_sort::node_t *n = &node_data.nodes[node_idx];

        float coord = n->X;
        if (!isfinite(coord)) {
            std::cout << "WARNING: invalid coordiate" << std::endl;
            coord = 0.0;
        }
        X[node_idx] = coord;
        //std::cout << "coords of " << node_idx << ": [" << X[node_idx*2] << "; " << Y[node_idx*2] << "] ; [" << X[node_idx*2+1] << "; " << Y[node_idx*2+1] <<"]\n";
    }


    // get rid of CUDA data structures
    free(node_data.nodes);
    for (int i = 0; i < path_count; i++) {
        free(path_data.paths[i].elements);
    }
    free(path_data.paths);
    free(zetas);


#ifdef cuda_layout_profiling
    auto end = std::chrono::high_resolution_clock::now();
    uint32_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CUDA layout took " << duration_ms << "ms" << std::endl;
#endif

    return X;
}

}
