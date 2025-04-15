#ifndef PROXYPOINT2D_H
#define PROXYPOINT2D_H

#include <Eigen/Core>
#include <spdlog/spdlog.h>
#include "parameters_sim.h"

struct ProxyPoint
{
    constexpr static unsigned nArrays = icy::SimParams::nPtsArrays;  // count of arrays in SOA
    bool isReference = false;
    unsigned pos, pitch;    // element # and capacity of each array in SOA
    double *soa;            // reference to SOA (assume contiguous space of size nArrays*pitch)
    double data[nArrays];    // local copy of the data when isReference==true

    ProxyPoint() { isReference = false; }

    ProxyPoint(const ProxyPoint &other);
    ProxyPoint& operator=(const ProxyPoint &other);

    // access data
    double getValue(size_t valueIdx) const;   // valueIdx < nArrays
    void setValue(size_t valueIdx, double value);
    Eigen::Vector2d getPos() const;
    Eigen::Vector2d getVelocity() const;
    bool getCrushedStatus();
    bool getDisabledStatus();
    bool getLiquidStatus();
    void setLiquidStatus(bool val);
    uint16_t getGrain();
    int getCellIndex(double hinv, unsigned GridY);  // index of the grid cell at the point's location
    int getXIndex(double hinv) const;                     // x-index of the grid cell
    void setPartition(uint8_t PartitionID);
    uint8_t getPartition();
};

#endif
