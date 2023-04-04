#include "odgi.hpp"
#include "algorithms/layout.hpp"
#include <omp.h>
#include "algorithms/xp.hpp"

#include "subcommand.hpp"
#include "args.hxx"
#include "utils.hpp"

#include <iostream>

namespace odgi {
    namespace algorithms {

        void compute_stress(const handlegraph::PathHandleGraph &graph, odgi::algorithms::layout::Layout &layout, 
                xp::XP &path_index, int thread_count, int64_t radius_arg, double &stress_result) {

            uint32_t max_step_count = 0;
            graph.for_each_path_handle(
                    [&] (const odgi::path_handle_t& p) {
                    uint32_t step_count = graph.get_step_count(p);
                    if (step_count > max_step_count) {
                        max_step_count = step_count;
                    }
                    });
            double sum_stress = 0;

            uint32_t path_count = path_index.path_count;
            uint32_t radius_max = (radius_arg >= 0)? uint32_t(radius_arg) : max_step_count;

            uint32_t total_step_count = 0;
            for (uint32_t radius = 0; radius <= radius_max; radius++) {
                double cur_radius_stress = 0.0;
#pragma omp parallel for schedule(dynamic) num_threads(thread_count)
                for (uint32_t pidx = 1; pidx <= path_count; pidx++) {
                    double path_stress = 0.0;
                    path_handle_t path = as_path_handle(pidx);
                    uint32_t path_step_count = path_index.get_path_step_count(path);
                    for (uint32_t rank = 0; rank < path_step_count; rank++) {   // < path_step_count - radius?
                        step_handle_t step_a;
                        as_integers(step_a)[0] = pidx;
                        as_integers(step_a)[1] = rank;
                        handle_t handle_a = path_index.get_handle_of_step(step_a);
                        // only analyze length of node itself
                        if (radius == 0) {
                            // path distance
                            double path_dist = double(graph.get_length(handle_a));
                            // visualization distance
                            odgi::algorithms::xy_d_t h_coords_start, h_coords_end;
                            h_coords_start = layout.coords(handle_a);
                            h_coords_end = layout.coords(graph.flip(handle_a));
                            double vis_dist = abs(odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end));

                            if (path_dist < 1.0) {
                                std::cerr << "path_dist (" << path_dist << ") smaller than 1.0" << std::endl;
                            }
                            if (path_dist <= 0.0) {
                                std::cerr << "path_dist (" << path_dist << ") smaller/equal than 0.0" << std::endl;
                                path_dist = 1e-9;
                            }
                            double stress_term = pow((path_dist - vis_dist) / path_dist, 2);
                            path_stress += stress_term;
                        } else {
                            if (rank < path_step_count - radius) {
                                step_handle_t step_b;
                                as_integers(step_b)[0] = pidx;
                                as_integers(step_b)[1] = rank + radius;
                                handle_t handle_b = path_index.get_handle_of_step(step_b);

                                double seq_len_a = double(graph.get_length(handle_a));
                                double seq_len_b = double(graph.get_length(handle_b));

                                double pos_in_path_a = double(path_index.get_position_of_step(step_a));
                                double pos_in_path_b = double(path_index.get_position_of_step(step_b));

                                for (int comb = 0; comb < 4; comb++) {
                                    double pos_a = pos_in_path_a;
                                    double pos_b = pos_in_path_b;

                                    bool node_a_is_reverse = graph.get_is_reverse(handle_a);
                                    bool use_other_end_a = node_a_is_reverse;
                                    if (comb > 1) {
                                        pos_a += seq_len_a;
                                        use_other_end_a = !node_a_is_reverse;
                                    }
                                    bool node_b_is_reverse = graph.get_is_reverse(handle_b);
                                    bool use_other_end_b = node_b_is_reverse;
                                    if (comb == 1 || comb == 3) {
                                        pos_b += seq_len_b;
                                        use_other_end_b = !node_b_is_reverse;
                                    }
                                    double path_dist = double(abs(pos_b - pos_a));

                                    // when radius 1: don't compute stress for direct neighbors (vis_dist should be close to 0.0)
                                    // also, when looping, don't compute for same elements
                                    if (path_dist > 0.0) {
                                        odgi::algorithms::xy_d_t vis_coords_a, vis_coords_b;
                                        if (!use_other_end_a) {
                                            vis_coords_a = layout.coords(handle_a);
                                        } else {
                                            vis_coords_a = layout.coords(graph.flip(handle_a));
                                        }
                                        if (!use_other_end_b) {
                                            vis_coords_b = layout.coords(handle_b);
                                        } else {
                                            vis_coords_b = layout.coords(graph.flip(handle_b));
                                        }
                                        double vis_dist = abs(double(odgi::algorithms::layout::coord_dist(vis_coords_a, vis_coords_b)));


                                        // TODO: why fails at radius greater 1?
                                        if (path_dist < 1.0) {
                                            std::cerr << "path_dist (" << path_dist << ") smaller than 1.0" << std::endl;
                                            std::cerr << "combination: " << comb << " path " << as_integers(step_a)[0];
                                            std::cerr << " a: id " << graph.get_id(handle_a) << " rank " << as_integers(step_a)[1] << " pos " << pos_in_path_a << " use-other-end " << use_other_end_a << " reverse " << node_a_is_reverse << " seq-len " << seq_len_a;
                                            std::cerr << " b: id " << graph.get_id(handle_b) << " rank " << as_integers(step_b)[1] << " pos " << pos_in_path_b << " use-other-end " << use_other_end_b << " reverse " << node_b_is_reverse << " seq-len " << seq_len_b;
                                            std::cerr << std::endl;
                                        }
                                        if (path_dist <= 0.0) {
                                            //std::cerr << "path_dist (" << path_dist << ") smaller/equal than 0.0" << std::endl;
                                            path_dist = 1e-9;
                                        }
                                        double stress_term = pow((path_dist - vis_dist) / path_dist, 2);
                                        path_stress += stress_term;
                                    } else {
                                        if (path_dist > 0.0) {
                                            std::cerr << "path_dist of " << path_dist << " should not be ignored" << std::endl;
                                        }
                                    }
                                }
                            }
                        }

                    }
                    //std::cout << "pidx: " << pidx << " step_count " << path_step_count << std::endl;
                    //total_step_count += path_step_count;

                    cur_radius_stress += path_stress;
                }
                sum_stress += cur_radius_stress;
                std::cout << "radius[" << radius << "/" << max_step_count << "]: added stress " << cur_radius_stress << " (total: " << sum_stress << ")" << std::endl;
            }
            stress_result = sum_stress;   // TODO remove
            std::cout << "total-step-count " << total_step_count << std::endl;




            /*
            std::vector<odgi::path_handle_t> paths;
            graph.for_each_path_handle([&] (const odgi::path_handle_t &p) {
                    paths.push_back(p);
                    });

            // std::cout << "TEST\tPANGENOME\tMODE" << std::endl;

            double sum_stress_squared_dist_weight = 0;
            uint32_t num_steps_iterated = 0;

            #pragma omp parallel for schedule(static, 1) num_threads(thread_count)
            for (auto p: paths) {
                double path_layout_dist;
                uint64_t path_nuc_dist;
                graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
                    path_layout_dist = 0;
                    path_nuc_dist = 0;
                    odgi::handle_t h = graph.get_handle_of_step(s);
                    odgi::algorithms::xy_d_t h_coords_start;
                    odgi::algorithms::xy_d_t h_coords_end;
                    if (graph.get_is_reverse(h)) {
                        h_coords_start = layout.coords(graph.flip(h));
                        h_coords_end = layout.coords(h);
                    } else {
                        h_coords_start = layout.coords(h);
                        h_coords_end = layout.coords(graph.flip(h));
                    }
                    // TODO refactor into function start
                    // did we hit the first step?
                    if (graph.has_previous_step(s)) {
                        odgi::step_handle_t prev_s = graph.get_previous_step(s);
                        odgi::handle_t prev_h = graph.get_handle_of_step(prev_s);
                        odgi::algorithms::xy_d_t prev_h_coords_start;
                        odgi::algorithms::xy_d_t prev_h_coords_end;
                        if (graph.get_is_reverse(prev_h)) {
                            prev_h_coords_start = layout.coords(graph.flip(prev_h));
                            prev_h_coords_end = layout.coords(prev_h);
                        } else {
                            prev_h_coords_start = layout.coords(prev_h);
                            prev_h_coords_end = layout.coords(graph.flip(prev_h));
                        }
                        double within_node_dist = 0;
                        double from_node_to_node_dist = 0;
                        if (!graph.get_is_reverse(prev_h)) {
                            /// f + f
                            if (!graph.get_is_reverse(h)) {
                                within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
                                from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_end, h_coords_start);
                            } else {
                                /// f + r
                                within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
                                from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_end, h_coords_end);
                            }
                        } else {
                            /// r + r
                            if (graph.get_is_reverse(h)) {
                                within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_end, h_coords_start);
                                from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_start, h_coords_end);
                            } else {
                                /// r + f
                                within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_end, h_coords_start);
                                from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_start, h_coords_start);
                            }
                        }
                        path_layout_dist += within_node_dist;
                        path_layout_dist += from_node_to_node_dist;
                        uint64_t nuc_dist = graph.get_length(h);
                        path_nuc_dist += nuc_dist;
                        // cur_window_end += nuc_dist;
                    } else {
                        // we only take a look at the current node
                        /// f
                        if (!graph.get_is_reverse(h)) {
                            path_layout_dist += odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
                        } else {
                            /// r
                            path_layout_dist += odgi::algorithms::layout::coord_dist(h_coords_end, h_coords_start);
                        }
                        uint64_t nuc_dist = graph.get_length(h);
                        path_nuc_dist += nuc_dist;
                        // cur_window_end += nuc_dist;
                    } // TODO refactor into function end

                    sum_stress_squared_dist_weight += pow((((double)path_layout_dist - (double)path_nuc_dist) / (double)path_nuc_dist), 2); // weight = 1 / (d*d)

                    num_steps_iterated += 1;
                });
            }

            stress_result = sum_stress_squared_dist_weight / (double)num_steps_iterated;
            */
        }
    }


    using namespace odgi::subcommand;

    int main_stress(int argc, char **argv) {

        // trick argumentparser to do the right thing with the subcommand
        for (uint64_t i = 1; i < argc - 1; ++i) {
            argv[i] = argv[i + 1];
        }
        std::string prog_name = "odgi stress";
        argv[0] = (char *) prog_name.c_str();
        --argc;


        args::ArgumentParser parser(
                "Compute stress of 2D layout.");
        args::Group mandatory_opts(parser, "[ MANDATORY OPTIONS ]");
        args::ValueFlag<std::string> dg_in_file(mandatory_opts, "FILE", "Load the succinct variation graph in ODGI format from this *FILE*. The file name usually ends with *.og*. It also accepts GFAv1, but the on-the-fly conversion to the ODGI format requires additional time!", {'i', "idx"});
        args::ValueFlag<std::string> layout_in_file(mandatory_opts, "FILE", "Read the layout coordinates from this .lay format FILE produced by odgi layout.", {'c', "coords-in"});
        args::ValueFlag<std::string> xp_in_file(mandatory_opts, "FILE", "Load the path index from this FILE so that it does not have to be created for the layout calculation.", {'X', "path-index"});
        args::ValueFlag<int64_t> radius_arg(mandatory_opts, "N", "Radius of stress analysis (when negative uses longest path length)", {'R', "radius"});

        args::Group threading_opts(parser, "[ Threading ]");
        args::ValueFlag<uint64_t> nthreads(threading_opts, "N",
                "Number of threads to use for parallel operations.",
                {'t', "threads"});

        args::Group processing_info_opts(parser, "[ Processsing Information ]");
        args::Flag progress(processing_info_opts, "progress", "Write the current progress to stderr.", {'P', "progress"});

        args::Group program_info_opts(parser, "[ Program Information ]");
        args::HelpFlag help(program_info_opts, "help", "Print a help summary for odgi stress.", {'h', "help"});

        try {
            parser.ParseCLI(argc, argv);
        } catch (args::Help) {
            std::cout << parser;
            return 0;
        } catch (args::ParseError e) {
            std::cerr << e.what() << std::endl;
            std::cerr << parser;
            return 1;
        }
        if (argc == 1) {
            std::cout << parser;
            return 1;
        }


        if (!dg_in_file) {
            std::cerr
                << "[odgi::stress] error: please specify an input file from where to load the graph via -i=[FILE], --idx=[FILE]."
                << std::endl;
            return 1;
        }

        if (!layout_in_file) {
            std::cerr
                << "[odgi::stress] error: please specify an input file from where to load the layout from via -c=[FILE], --coords-in=[FILE]."
                << std::endl;
            return 1;
        }

        if (!xp_in_file) {
            std::cerr
                << "[odgi::stress] error: please specify an input file from where to load the path-index from via -X=[FILE], --path-index=[FILE]."
                << std::endl;
            return 1;
        }

        const uint64_t num_threads = nthreads ? args::get(nthreads) : 1;
        const int64_t radius = radius_arg ? args::get(radius_arg) : 1;

        graph_t graph;
        if (!args::get(dg_in_file).empty()) {
            std::string infile = args::get(dg_in_file);
            if (infile == "-") {
                graph.deserialize(std::cin);
            } else {
                utils::handle_gfa_odgi_input(infile, "layout", args::get(progress), num_threads, graph);
            }
        }

        algorithms::layout::Layout layout;
        if (layout_in_file) {
            auto& infile = args::get(layout_in_file);
            if (!infile.empty()) {
                if (infile == "-") {
                    layout.load(std::cin);
                } else {
                    ifstream f(infile.c_str());
                    layout.load(f);
                    f.close();
                }
            }
        }

        xp::XP path_index;
        if (xp_in_file) {
            std::ifstream in;
            in.open(args::get(xp_in_file));
            path_index.load(in);
            in.close();
        }

        double stress_result = 0;
        algorithms::compute_stress(graph, layout, path_index, num_threads, radius, stress_result);
        std::cout << "Stress: " << stress_result << std::endl;
        return 0;
    }

    static Subcommand odgi_stress("stress", "Compute stress of 2D layouts.", PIPELINE, 3, main_stress);

}
