#ifndef BVHN2D_H
#define BVHN2D_H

#include <vector>
#include <utility>
#include <Eigen/Core>

#include "kdop8.h"
#include "SimplePool.h"


class BVHN2D
{
public:
    static SimplePool<std::vector<BVHN2D*>> VectorFactory;
    static SimplePool<BVHN2D> BVHNFactory;

    kDOP8 box;
    int elem; // element that is enveloped by this kDOP, if leaf
    BVHN2D *child1, *child2;

    bool isLeaf = false;
    uint8_t level;

    void Build(std::vector<BVHN2D*> &bvs, const uint8_t level_);
    void Expand(Eigen::Vector2f v);
    void Collide(BVHN2D *b, std::vector<std::pair<BVHN2D*,BVHN2D*>> &broad_list);
};

#endif
