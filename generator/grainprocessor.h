#ifndef GRAINPROCESSOR_H
#define GRAINPROCESSOR_H

#include <string>
#include <vector>
#include <array>

#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <gmsh.h>

#include "bvhn.h"

struct Tetra
{
    Eigen::Vector3f nds[4];
    int grain;
};


class GrainProcessor
{

public:
    void LoadMSH(std::string fileName);
    void IdentifyGrains(const float scale);
    void Write_HDF5(std::string fileName);
    static bool PointInsideTetrahedron(Eigen::Vector3f point, Eigen::Vector3f tetra[4]);
    static std::vector<std::array<float, 3>> GenerateBlock(float dx, float dy, float dz, int n);

    void generate_block(float bx, float by, float bz, int n);
    void generate_cone(float diameter, float top, float angle, float height, int n);

    std::vector<std::array<float, 3>> buffer;
    std::vector<short> grainID;

private:
    float volume = -1;
    std::vector<BVHN*> leaves;
    BVHN root;

    std::vector<Eigen::Vector3f> vertices, vertices2;
    std::vector<std::array<int,5>> elems, elems2;   // 4 nodes + grain id
    std::vector<Tetra> tetra, tetra2;

};

#endif // GRAINPROCESSOR_H
