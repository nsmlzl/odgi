#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>
#include <random>
#include <math.h>
//#include <curand.h>
//#include <curand_kernel.h>
#include <sstream>
#include <iomanip>

#include "odgi.hpp"
#include "xp.hpp"
#include "XoshiroCpp.hpp"
#include "dirty_zipfian_int_distribution.h"


#define cuda_layout_profiling


namespace extended_sort {


struct node_t {
    //int32_t seq_length;
    double X;
};
struct node_data_t {
    uint32_t node_count;
    node_t *nodes;
};


struct path_element_t {
    uint32_t node_id;
    int64_t pos;    // if position negative: reverse orientation
};

struct path_t {
    uint32_t step_count;
    path_element_t *elements;
};

struct path_data_t {
    uint32_t path_count;
    uint64_t total_path_steps;
    path_t *paths;
};


// TODO: make parameters constant?
struct layout_config_t {
    uint64_t iter_max;
    uint64_t min_term_updates;
    double eta_max;
    double eps;
    int32_t iter_with_max_learning_rate;
    uint32_t first_cooling_iteration;
    double theta;
    uint64_t space;
    uint64_t space_max;
    uint64_t space_quantization_step;
    int nthreads;
};


std::vector<double> cache_optimized_sort(layout_config_t config, const odgi::graph_t &graph, const xp::XP &path_index);

}
