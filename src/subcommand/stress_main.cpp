#include "odgi.hpp"
#include "algorithms/layout.hpp"
#include <omp.h>

#include "subcommand.hpp"
#include "args.hxx"
#include "utils.hpp"

#include <iostream>

namespace odgi {
    namespace algorithms {
        // function to compute stress, given the current layout
        // This is copied from previous implementation on the original odgi implementation. 
        // We can design easier way to compute metrics using the lean data structure. [TODO]
        void compute_stress(odgi::algorithms::layout::Layout &layout, 
                const handlegraph::PathHandleGraph &graph, 
                int thread_count,
                double &stress_result) {

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

        const uint64_t num_threads = nthreads ? args::get(nthreads) : 1;

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

        double stress_result = 0;
        algorithms::compute_stress(layout, graph, num_threads, stress_result);
        std::cout << "Stress: " << stress_result << std::endl;
        return 0;
    }

    static Subcommand odgi_stress("stress", "Compute stress of 2D layouts.", PIPELINE, 3, main_stress);

}
