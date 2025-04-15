#include "host_side_soa.h"



std::pair<Eigen::Vector2d, Eigen::Vector2d> HostSideSOA::getBlockDimensions()
{
    Eigen::Vector2d result[2];
    for(int k=0;k<2;k++)
    {
        std::pair<SOAIterator, SOAIterator> it_res = std::minmax_element(begin(), end(),
                                                                         [k](ProxyPoint &p1, ProxyPoint &p2)
                                                                         {return p1.getValue(icy::SimParams::posx+k)<p2.getValue(icy::SimParams::posx+k);});
        result[0][k] = (*it_res.first).getValue(icy::SimParams::posx+k);
        result[1][k] = (*it_res.second).getValue(icy::SimParams::posx+k);
    }
    return {result[0], result[1]};
}

void HostSideSOA::offsetBlock(Eigen::Vector2d offset)
{
    for(SOAIterator it = begin(); it!=end(); ++it)
    {
        ProxyPoint &p = *it;
        Eigen::Vector2d pos = p.getPos();
        pos += offset;
        p.setValue(icy::SimParams::posx, pos.x());
        p.setValue(icy::SimParams::posx+1, pos.y());
    }
}

void HostSideSOA::addVelocityForTesting()
{
    for(SOAIterator it = begin(); it!=end(); ++it)
    {
        ProxyPoint &p = *it;
        bool isLiquid = p.getLiquidStatus();
        if(!isLiquid)
        {
            Eigen::Vector2d pos = p.getPos();
            double vely = -2*(pos.x()-2);
            double velx = 3;
            p.setValue(icy::SimParams::velx, velx);
            p.setValue(icy::SimParams::velx+1, vely);
        }
    }
}



void HostSideSOA::RemoveDisabledAndSort(double hinv, unsigned GridY)
{
    spdlog::info("RemoveDisabledAndSort; nPtsArrays {}", icy::SimParams::nPtsArrays);
    unsigned size_before = size;
    SOAIterator it_result = std::remove_if(begin(), end(), [](ProxyPoint &p){return p.getDisabledStatus();});
    size = it_result.m_point.pos;
    spdlog::info("RemoveDisabledAndSort: {} removed; new size {}", size_before-size, size);
    std::sort(begin(), end(),
              [&hinv,&GridY](ProxyPoint &p1, ProxyPoint &p2)
              {return p1.getCellIndex(hinv,GridY)<p2.getCellIndex(hinv,GridY);});
    spdlog::info("RemoveDisabledAndSort done");
}


void HostSideSOA::Allocate(unsigned capacity)
{
    cudaFreeHost(host_buffer);
    this->capacity = capacity;
    size_t allocation_size = sizeof(double)*capacity*icy::SimParams::nPtsArrays;
    cudaError_t err = cudaMallocHost(&host_buffer, allocation_size);
    if(err != cudaSuccess)
    {
        const char *description = cudaGetErrorString(err);
        spdlog::critical("allocating host buffer of size {}: {}",allocation_size,description);
        throw std::runtime_error("allocating host buffer for points");
    }
    size = 0;
    memset(host_buffer, 0, allocation_size);
    spdlog::info("HSSOA allocate capacity {} pt; toal {} Gb", capacity, (double)allocation_size/(1024.*1024.*1024.));
}

unsigned HostSideSOA::FindFirstPointAtGridXIndex(const int index_grid_x, const double hinv)
{
    SOAIterator it = std::lower_bound(begin(),end(),index_grid_x,
                                      [hinv](const ProxyPoint &p, const int val)
                                      {return p.getXIndex(hinv)<val;});

    unsigned result_pos = it.m_point.pos;
//    int xindex = it.m_point.getXIndex(hinv);
//    spdlog::info("FindFirstPointAtGridXIndex: index_grid_x {} at pos {}; found cell_index {}", index_grid_x, result_pos, xindex);
    return result_pos;
}


void HostSideSOA::InitializeBlock()
{
    Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
    for(SOAIterator it = begin(); it!=end(); ++it)
    {
        ProxyPoint &p = *it;
        p.setValue(icy::SimParams::idx_Jp_inv,1);
        for(int i=0; i<icy::SimParams::dim; i++)
                p.setValue(icy::SimParams::Fe00+i*2+i, 1.0);
    }

}




// ==================================================== SOAIterator

SOAIterator::SOAIterator(unsigned pos, double *soa_data, unsigned pitch)
{
    m_point.isReference = true;
    m_point.pos = pos;
    m_point.soa = soa_data;
    m_point.pitch = pitch;
}

SOAIterator::SOAIterator(const SOAIterator& other)
{
    m_point.isReference = other.m_point.isReference;
    m_point.pos = other.m_point.pos;
    m_point.soa = other.m_point.soa;
    m_point.pitch = other.m_point.pitch;
}

SOAIterator& SOAIterator::operator=(const SOAIterator& other)
{
    m_point.isReference = other.m_point.isReference;
    m_point.pos = other.m_point.pos;
    m_point.soa = other.m_point.soa;
    m_point.pitch = other.m_point.pitch;
    return *this;
}

