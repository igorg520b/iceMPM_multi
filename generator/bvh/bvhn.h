#ifndef BVHN_H
#define BVHN_H

#include <vector>
#include <Eigen/Core>

#include "kdop24.h"
#include "SimplePool.h"

struct BVHN
{
public:
    static SimplePool<std::vector<BVHN*>> VectorFactory;
    static SimplePool<BVHN> BVHNFactory;

    kDOP24 box;
    int elem; // element that is enveloped by this kDOP, if leaf
    BVHN *child1, *child2;

    bool isLeaf = false;
    uint8_t level;

    void Build(std::vector<BVHN*> &bvs, const uint8_t level_);
    void Expand(Eigen::Vector3f v);
    void Collide(BVHN *b, std::vector<std::pair<BVHN*,BVHN*>> &broad_list);
};

#endif // BVHN_H
