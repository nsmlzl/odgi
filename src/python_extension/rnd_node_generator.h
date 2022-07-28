#pragma once

#include "odgi.hpp"
#include "xp.hpp"
#include "XoshiroCpp.hpp"
#include "dirty_zipfian_int_distribution.h"

#include <vector>
#include <thread>

namespace python_extension {

    struct random_nodes_pack_t {
        uint64_t id_n0;
        uint64_t id_n1;
        uint32_t vis_p_n0;      // use visualization point 0 or 1
        uint32_t vis_p_n1;      // use visualization point 0 or 1
        double distance;
    };

    class RndNodeGenerator {
        public:
        RndNodeGenerator(odgi::graph_t &graph, double zipf_theta, uint64_t space_max, uint64_t space_quantization_step);
        ~RndNodeGenerator();

        random_nodes_pack_t get_random_node_pack(bool cooling);
        void get_random_node_batch(int batch_size, int64_t *i, int64_t *j, int64_t *vis_i, int64_t *vis_j, double *d, bool cooling, int nthreads);
        uint64_t get_max_path_length(void);

        private:
        odgi::graph_t &_graph;
        xp::XP _path_index;
        const sdsl::int_vector<> *_nr_iv;
        const sdsl::int_vector<> *_npi_iv;

        double _theta;
        uint64_t _space;
        uint64_t _space_max;
        uint64_t _space_quantization_step;
        std::vector<double> _zetas;

        XoshiroCpp::Xoshiro256Plus _rng_gen;
        std::uniform_int_distribution<uint64_t> _dis_step;
        std::uniform_int_distribution<uint64_t> _flip;
    };


}
