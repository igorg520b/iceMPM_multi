#ifndef GRAINPROCESSOR2D_H
#define GRAINPROCESSOR2D_H

#include <Eigen/Core>
#include "bvhn2d.h"

struct Triangle
{
    Eigen::Vector2f nds[3];
    int grain;
};


class GrainProcessor2D
{
    constexpr static long long status_liquid = 0x40000ll;
    constexpr static float waterLevel = 0.6;


public:
    void generate_block_and_write(float scale, float bx, float by, int n, std::string msh, std::string outputFile);
    void generate_floe_and_write(float scale, float bx, float by, int n, int grid, std::string msh, std::string outputFile);

    std::vector<std::array<float, 2>> buffer;   // result from poisson disk sampler
    std::vector<short> grainID;

    // SOA format (to be written as HDF5)
    std::vector<uint64_t> llGrainID;
    std::vector<double> coordinates[2];

    // mesh with grains
    std::vector<Eigen::Vector2f> vertices2;
    std::vector<std::array<int,4>> elems2;   // 4 nodes + grain id
    std::vector<Triangle> tris2;


private:
    void GenerateBlock(float dx, float dy, int n);
    void GenerateFloe(float dx, float dy, int n, int grid);
    void LoadMSH(std::string fileName);
    void IdentifyGrains(const float scale);
    void Write_HDF5(std::string fileName, int OffsetIncluded = 0);
    static bool PointInsideTriangle(Eigen::Vector2f point, Eigen::Vector2f triangle[3]);

    float volume = -1;
    std::vector<BVHN2D*> leaves;
    BVHN2D root;
};

#endif // GRAINPROCESSOR2D_H
