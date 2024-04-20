#include <stdexcept>
#include <cfloat>
#include <algorithm>
#include <iostream>
#include "bvhn2d.h"

SimplePool<std::vector<BVHN2D*>> BVHN2D::VectorFactory(50);
SimplePool<BVHN2D> BVHN2D::BVHNFactory(100000);


void BVHN2D::Build(std::vector<BVHN2D*> &bvs, const uint8_t level_)
{
    // TODO: ensure that the algorithm will also work with just one element in bvs

    if(level_ > 100) throw std::runtime_error("BVH level is over 100");
    level = level_;
    std::size_t count = bvs.size();
    if(count == 0) throw std::runtime_error("bvs->size==0 in icy::BVHN::Build");
    else if(count == 1) throw std::runtime_error("bvs->size==1 in icy::BVHN::Build");

    isLeaf = false;

    box.Reset();
    for(auto const &bv : bvs) box.Expand(bv->box); // expand box to the size of bvs collection

    std::vector<BVHN2D*> *left = VectorFactory.take();
    std::vector<BVHN2D*> *right = VectorFactory.take();
    left->clear();
    right->clear();

    std::vector<BVHN2D*>::iterator iter;
    if (box.dX >= box.dY)
    {
        float ctrX = box.ctrX;
        iter = std::partition(bvs.begin(),bvs.end(),[ctrX](const BVHN2D *bv){return bv->box.ctrX < ctrX;});
        left->resize(std::distance(bvs.begin(),iter));
        right->resize(std::distance(iter,bvs.end()));
        std::copy(bvs.begin(),iter,left->begin());
        std::copy(iter,bvs.end(),right->begin());
        // ensure that there is at least one element on each side
        if(left->size() == 0)
        {
            auto iter = std::min_element(right->begin(), right->end(),
                                         [](BVHN2D* b1, BVHN2D* b2) {return b1->box.ctrX < b2->box.ctrX;});
            // move "selected" from left to right
            left->push_back(*iter);
            right->erase(iter);
        }
        else if(right->size() == 0)
        {
            auto iter = std::max_element(left->begin(), left->end(),
                                         [](BVHN2D* b1, BVHN2D* b2) {return b1->box.ctrX < b2->box.ctrX;});
            // move selected from left to right
            right->push_back(*iter);
            left->erase(iter);
        }
    }
    else
    {
        float ctr = box.ctrY;
        iter = std::partition(bvs.begin(),bvs.end(),[ctr](const BVHN2D *bv){return bv->box.ctrY < ctr;});
        left->resize(std::distance(bvs.begin(),iter));
        right->resize(std::distance(iter,bvs.end()));
        std::copy(bvs.begin(),iter,left->begin());
        std::copy(iter,bvs.end(),right->begin());
        if(left->size() == 0)
        {
            auto iter = std::min_element(right->begin(), right->end(),
                                         [](BVHN2D* b1, BVHN2D* b2) {return b1->box.ctrY < b2->box.ctrY;});
            // move "selected" from left to right
            left->push_back(*iter);
            right->erase(iter);
        }
        else if(right->size() == 0)
        {
            auto iter = std::max_element(left->begin(), left->end(),
                                         [](BVHN2D* b1, BVHN2D* b2) {return b1->box.ctrY < b2->box.ctrY;});
            // move selected from left to right
            right->push_back(*iter);
            left->erase(iter);
        }
    }


    if(left->size() == 1)
    {
        child1 = left->front();
        child1->level=level+1;
    }
    else if(left->size() > 1)
    {
        child1 = BVHNFactory.take();
        child1->Build(*left, level+1);
    }
    else throw std::runtime_error("left.size < 1");
    VectorFactory.release(left);

    if(right->size() == 1)
    {
        child2 = right->front();
        child2->level=level+1;
    }
    else if(right->size() > 1)
    {
        child2 = BVHNFactory.take();
        child2->Build(*right, level+1);
    }
    else throw std::runtime_error("right.size < 1");
    VectorFactory.release(right);
}


void BVHN2D::Expand(Eigen::Vector2f v)
{
    if(!isLeaf) throw std::runtime_error("Expand: not leaf");
    box.Expand(v[0],v[1]);
}


void BVHN2D::Collide(BVHN2D *b, std::vector<std::pair<BVHN2D*,BVHN2D*>> &broad_list)
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


