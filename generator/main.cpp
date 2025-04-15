#include <cxxopts.hpp>

#include <gmsh.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "2d/grainprocessor2d.h"

// -s 2d -n 250000 -o 2d_250k.h5 -m /home/s2/Projects/GrainIdentifier/msh_2d/1k_2d.msh -c 2
// -s floe -n 800000 -o floe_800k.h5 -m /home/s2/Projects-CUDA/iceMPM_multi/generator/msh_2d/1k_2d.msh -c 2 -x 4 -y 1 -g 512


int main(int argc, char *argv[])
{
    gmsh::initialize();
    spdlog::info("testing threads {}", omp_get_max_threads());
#pragma omp parallel
    {     spdlog::info("{}", omp_get_thread_num()); }
    std::cout << std::endl;


    // parse options
    cxxopts::Options options("grain identifier", "Generate raw input file for MPM simulation");

    options.add_options()
        // point generation
        ("s,shape", "Shape to generate (cone, block, 2d)", cxxopts::value<std::string>()->default_value("cone"))
        ("n,numberofpoints", "Make a set of N points for the simulation starting input", cxxopts::value<int>()->default_value("10000000"))
        ("o,output", "Output file name", cxxopts::value<std::string>()->default_value("raw_10m.h5"))
        ("m,msh", "Input file with grains", cxxopts::value<std::string>())
        ("c,scale", "Scale for grain mapping", cxxopts::value<float>()->default_value("3"))

        // for block
        ("x,bx", "Length of the block", cxxopts::value<float>()->default_value("2.5"))
        ("y,by", "Height of the block", cxxopts::value<float>()->default_value("1.0"))
        ("g,grid", "Grid size in x-direction", cxxopts::value<int>()->default_value("512"))
        ;

    auto option_parse_result = options.parse(argc, argv);


    // generate points input file
    std::string shape = option_parse_result["shape"].as<std::string>();
    int n = option_parse_result["numberofpoints"].as<int>();
    std::string output_file = option_parse_result["output"].as<std::string>();
    std::string msh_file = option_parse_result["msh"].as<std::string>();
    float scale = option_parse_result["scale"].as<float>();
    int grid = option_parse_result["grid"].as<int>();

    float bx = option_parse_result["bx"].as<float>();
    float by = option_parse_result["by"].as<float>();

    GrainProcessor2D gp2d;

    if(shape == "2d")
    {
        gp2d.generate_block_and_write(scale, bx, by, n, msh_file, output_file);
    }
    else if(shape == "floe")
    {
        gp2d.generate_floe_and_write(scale, bx, by, n, grid, msh_file, output_file);

    }
    else throw std::runtime_error("incorrect shape");
}
