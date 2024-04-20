#include <stdexcept>
#include <cfloat>
#include <algorithm>
#include "bvhn.h"
#include <spdlog/spdlog.h>

SimplePool<std::vector<BVHN*>> BVHN::VectorFactory(50);
SimplePool<BVHN> BVHN::BVHNFactory(1000000);


void BVHN::Build(std::vector<BVHN*> &bvs, const uint8_t level_)
{
    if(level_ > 100)
    {
        spdlog::critical("BVH level is {}; bvs size {}", level_, bvs.size());
        throw std::runtime_error("BVH level is over 100");
    }
    level = level_;
    std::size_t count = bvs.size();
    if(count == 0) throw new std::runtime_error("bvs->size==0 in BVHN::Initialize");
    else if(count == 1) throw new std::runtime_error("bvs->size==1 in BVHN::Initialize");

    isLeaf = false;

    box.Reset();
    for(auto const &bv : bvs) box.Expand(bv->box); // expand box to the size of bvs collection

    std::vector<BVHN*> *left = VectorFactory.take();
    std::vector<BVHN*> *right = VectorFactory.take();
    left->clear();
    right->clear();

    std::vector<BVHN*>::iterator iter;
    if (box.dX >= box.dY && box.dX >= box.dZ)
    {
        float ctrX = box.ctrX;

        iter = std::partition(bvs.begin(),bvs.end(),[ctrX](const BVHN *bv){return bv->box.ctrX < ctrX;});
        left->resize(std::distance(bvs.begin(),iter));
        right->resize(std::distance(iter,bvs.end()));
        std::copy(bvs.begin(),iter,left->begin());
        std::copy(iter,bvs.end(),right->begin());
        // ensure that there is at least one element on each side
        if(left->size() == 0)
        {
            auto iter = std::min_element(right->begin(), right->end(),
                                         [](BVHN* b1, BVHN* b2) {return b1->box.ctrX < b2->box.ctrX;});
            // move "selected" from left to right
            left->push_back(*iter);
            right->erase(iter);
        }
        else if(right->size() == 0)
        {
            auto iter = std::max_element(left->begin(), left->end(),
                                         [](BVHN* b1, BVHN* b2) {return b1->box.ctrX < b2->box.ctrX;});
            // move selected from left to right
            right->push_back(*iter);
            left->erase(iter);
        }
    }
    else if(box.dY >= box.dX && box.dY >= box.dZ)
    {
        float ctr = box.ctrY;
        iter = std::partition(bvs.begin(),bvs.end(),[ctr](const BVHN *bv){return bv->box.ctrY < ctr;});
        left->resize(std::distance(bvs.begin(),iter));
        right->resize(std::distance(iter,bvs.end()));
        std::copy(bvs.begin(),iter,left->begin());
        std::copy(iter,bvs.end(),right->begin());
        if(left->size() == 0)
        {
            auto iter = std::min_element(right->begin(), right->end(),
                                         [](BVHN* b1, BVHN* b2) {return b1->box.ctrY < b2->box.ctrY;});
            // move "selected" from left to right
            left->push_back(*iter);
            right->erase(iter);
        }
        else if(right->size() == 0)
        {
            auto iter = std::max_element(left->begin(), left->end(),
                                         [](BVHN* b1, BVHN* b2) {return b1->box.ctrY < b2->box.ctrY;});
            // move selected from left to right
            right->push_back(*iter);
            left->erase(iter);
        }
    }
    else
    {
        float ctr = box.ctrZ;
        iter = std::partition(bvs.begin(),bvs.end(),[ctr](const BVHN *bv){return bv->box.ctrZ < ctr;});
        left->resize(std::distance(bvs.begin(),iter));
        right->resize(std::distance(iter,bvs.end()));
        std::copy(bvs.begin(),iter,left->begin());
        std::copy(iter,bvs.end(),right->begin());
        if(left->size() == 0)
        {
            auto iter = std::min_element(right->begin(), right->end(),
                                         [](BVHN* b1, BVHN* b2) {return b1->box.ctrZ < b2->box.ctrZ;});
            // move "selected" from left to right
            left->push_back(*iter);
            right->erase(iter);
        }
        else if(right->size() == 0)
        {
            auto iter = std::max_element(left->begin(), left->end(),
                                         [](BVHN* b1, BVHN* b2) {return b1->box.ctrZ < b2->box.ctrZ;});
            // move selected from left to right
            right->push_back(*iter);
            left->erase(iter);
        }
    }

    //spdlog::info("lv {}; lf {}; r{}; bb {}; {}; {}", level, left->size(), right->size(), box.dX, box.dY, box.dZ);

    if(left->size() == 0 || right->size() == 0) throw std::runtime_error("BVHN build error");

    if(left->size() == 1)
    {
        child1 = left->front();
    }
    else
    {
        child1 = BVHNFactory.take();
        child1->Build(*left, level+1);
    }
    VectorFactory.release(left);

    if(right->size() == 1)
    {
        child2 = right->front();
    }
    else
    {
        child2 = BVHNFactory.take();
        child2->Build(*right, level+1);
    }

    VectorFactory.release(right);
}


void BVHN::Collide(BVHN *b, std::vector<std::pair<BVHN*,BVHN*>> &broad_list)
{
    if(!box.Overlaps(b->box)) return;
    if (this->isLeaf && b->isLeaf)
    {
        broad_list.emplace_back(this,b);
    }
    else if (this->isLeaf)
    {
        Collide(b->child1, broad_list);
        Collide(b->child2, broad_list);
    }
    else
    {
        b->Collide(child1, broad_list);
        b->Collide(child2, broad_list);
    }
}

void BVHN::Expand(Eigen::Vector3f v)
{
    if(!isLeaf) throw std::runtime_error("Expand: not leaf");
    box.Expand(v[0],v[1],v[2]);
}

