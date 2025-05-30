#ifndef HOSTSIDESOA_H
#define HOSTSIDESOA_H

#include <utility>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

#include <Eigen/Core>
#include <spdlog/spdlog.h>

#include "parameters_sim.h"
#include "proxypoint2d.h"



class SOAIterator
{
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = ProxyPoint;
    using difference_type = int;
    using pointer = ProxyPoint*;
    using reference = ProxyPoint&;

    ProxyPoint m_point;

    SOAIterator(unsigned pos, double *soa_data, unsigned pitch);
    SOAIterator(const SOAIterator& other);
    SOAIterator& operator=(const SOAIterator& other);
    SOAIterator() {};


    bool operator<(const SOAIterator& t) const {return (m_point.pos<t.m_point.pos && m_point.soa==t.m_point.soa);}
    bool operator==(const SOAIterator& t)const{return (m_point.pos == t.m_point.pos && m_point.soa==t.m_point.soa);}
    bool operator!=(const SOAIterator& t)const{return (m_point.pos != t.m_point.pos || m_point.soa!=t.m_point.soa);}

    SOAIterator& operator+=(difference_type n) { m_point.pos+=n; return (*this); }
    SOAIterator& operator-=(difference_type n) { m_point.pos-=n; return (*this); }
    SOAIterator& operator++() { ++m_point.pos; return (*this); }
    SOAIterator& operator--() { --m_point.pos; return (*this); }
    SOAIterator operator+(const difference_type& m) {SOAIterator r=*this;r.m_point.pos+=m;return r;}
    SOAIterator operator-(const difference_type& m) {SOAIterator r=*this;r.m_point.pos-=m;return r;}
    difference_type operator-(const SOAIterator& rawIterator){return m_point.pos-rawIterator.m_point.pos;}
    reference operator*() {return m_point;}
    pointer operator->() {return &m_point;}
};


// the class manages the Structure-of-Arrays (SOA) buffer
class HostSideSOA
{
public:
    double *host_buffer = nullptr; // buffer in page-locked memory for transferring the data between device and host
    unsigned capacity;  // max number of points that the host-side buffer can hold
    unsigned size = 0;      // the number of points, including "disabled" ones, in the host buffer (may fluctuate)

    SOAIterator begin(){return SOAIterator(0, host_buffer, capacity);}
    SOAIterator end(){return SOAIterator(size, host_buffer, capacity);}

    void Allocate(unsigned capacity);
    void RemoveDisabledAndSort(double hinv, unsigned GridY);
    unsigned FindFirstPointAtGridXIndex(const int index_grid_x, const double hinv);
    void InitializeBlock(); // set the matrices that are supposed to be identity, i.e. Fe

    double* getPointerToPosX() {return host_buffer + capacity*icy::SimParams::posx;}
    double* getPointerToPosY() {return host_buffer + capacity*(icy::SimParams::posx+1);}
    double* getPointerToLine(int idxLine) {return host_buffer + capacity*idxLine;}

    std::pair<Eigen::Vector2d, Eigen::Vector2d> getBlockDimensions();
    void offsetBlock(Eigen::Vector2d offset);
    void addVelocityForTesting();
};

#endif // HOSTSIDESOA_H
