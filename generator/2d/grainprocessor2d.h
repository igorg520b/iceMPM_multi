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
public:
    void generate_block_and_write(float scale, float bx, float by, int n, std::string msh, std::string outputFile);



    std::vector<std::array<float, 2>> buffer;
    std::vector<short> grainID;

    // SOA format - copies from buffer and GrainID
    std::vector<uint64_t> llGrainID;
    std::vector<double> coordinates[2];

    std::vector<Eigen::Vector2f> vertices2;
    std::vector<std::array<int,4>> elems2;   // 4 nodes + grain id
    std::vector<Triangle> tris2;


private:
    void GenerateBlock(float dx, float dy, int n);
    void LoadMSH(std::string fileName);
    void IdentifyGrains(const float scale);
    void Write_HDF5(std::string fileName);
    static bool PointInsideTriangle(Eigen::Vector2f point, Eigen::Vector2f triangle[3]);

    float volume = -1;
    std::vector<BVHN2D*> leaves;
    BVHN2D root;
};

#endif // GRAINPROCESSOR2D_H
