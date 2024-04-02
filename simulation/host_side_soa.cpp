#include "host_side_soa.h"



std::pair<Eigen::Vector2d, Eigen::Vector2d> HostSideSOA::getBlockDimensions()
{
    Eigen::Vector2d result[2];
    for(int k=0;k<2;k++)
    {
        std::pair<SOAIterator, SOAIterator> it_res = std::minmax_element(begin(), end(), [k](ProxyPoint &p1, ProxyPoint &p2) {return p1.getValue(icy::SimParams::posx+k)<p2.getValue(icy::SimParams::posx+k);});
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

void HostSideSOA::RemoveDisabledAndSort(double hinv, unsigned GridY)
{
    unsigned size_before = size;
    SOAIterator it_result = std::remove_if(begin(), end(), [](ProxyPoint &p){return p.getDisabledStatus();});
    size = it_result.m_point.pos;
    std::sort(begin(), end(),
              [&hinv,&GridY](ProxyPoint &p1, ProxyPoint &p2)
              {return p1.getCellIndex(hinv,GridY)<p2.getCellIndex(hinv,GridY);});
    spdlog::info("RemoveDisabledAndSort: {} removed", size_before-size);
}


void HostSideSOA::Allocate(unsigned capacity)
{
    cudaFreeHost(host_buffer);
    this->capacity = capacity;
    size_t allocation_size = sizeof(double)*capacity*icy::SimParams::nPtsArrays;
    cudaError_t err = cudaMallocHost(&host_buffer, allocation_size);
    if(err!=cudaSuccess) throw std::runtime_error("allocating host buffer for points");
    size = 0;
    memset(host_buffer, 0, allocation_size);
    spdlog::info("HSSOA allocate capacity {} pt; toal {} Gb", capacity, (double)allocation_size/(1024.*1024.*1024.));
}

unsigned HostSideSOA::FindFirstPointAtGridXIndex(int index_grid_x, double hinv)
{
    SOAIterator it = std::lower_bound(begin(),end(),index_grid_x,
                                      [hinv](const ProxyPoint &p, const float val)
                                      {return p.getXIndex(hinv)<val;});
    return it.m_point.pos;
}


void HostSideSOA::InitializeBlock()
{
    Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
    for(SOAIterator it = begin(); it!=end(); ++it)
    {
        ProxyPoint &p = *it;

        for(int i=0; i<icy::SimParams::dim; i++)
            for(int j=0; j<icy::SimParams::dim; j++)
                p.setValue(icy::SimParams::Fe00+i*2+j, identity(i,j));
    }

}


/*
void icy::Point::TransferToBuffer(double *buffer, const int pitch, const int point_index) const
{
    char* ptr_intact = (char*)(&buffer[pitch*icy::SimParams::idx_utility_data]);
    ptr_intact[point_index] = crushed;

    short* ptr_grain = (short*)(&ptr_intact[pitch]);
    ptr_grain[point_index] = grain;

    buffer[point_index + pitch*icy::SimParams::idx_Jp_inv] = Jp_inv;

    for(int i=0; i<icy::SimParams::dim; i++)
    {
        buffer[point_index + pitch*(icy::SimParams::posx+i)] = pos[i];
        buffer[point_index + pitch*(icy::SimParams::velx+i)] = velocity[i];
        for(int j=0; j<icy::SimParams::dim; j++)
        {
            buffer[point_index + pitch*(icy::SimParams::Fe00 + i*icy::SimParams::dim + j)] = Fe(i,j);
            buffer[point_index + pitch*(icy::SimParams::Bp00 + i*icy::SimParams::dim + j)] = Bp(i,j);
        }
    }
}
*/


// ====================================================== ProxyPoint
ProxyPoint::ProxyPoint(const ProxyPoint &other)
{
    isReference = false;
    // local copy
    if(other.isReference)
    {
        for(int i=0;i<nArrays;i++) data[i] = other.soa[other.pos + i*other.pitch];
    }
    else
    {
        for(int i=0;i<nArrays;i++) data[i] = other.data[i];
    }
}

ProxyPoint& ProxyPoint::operator=(const ProxyPoint &other)
{
    if(isReference)
    {
        // distribute into soa
        if(other.isReference)
        {
            for(int i=0;i<nArrays;i++) soa[pos + i*pitch] = other.soa[other.pos + i*other.pitch];
        }
        else
        {
            for(int i=0;i<nArrays;i++) soa[pos + i*pitch] = other.data[i];
        }
    }
    else
    {
        // local copy
        if(other.isReference)
        {
            for(int i=0;i<nArrays;i++) data[i] = other.soa[other.pos + i*other.pitch];
        }
        else
        {
            for(int i=0;i<nArrays;i++) data[i] = other.data[i];
        }
    }
    return *this;
}

Eigen::Vector2d ProxyPoint::getPos() const
{
    Eigen::Vector2d result;
    if(isReference)
    {
        for(int i=0; i<icy::SimParams::dim;i++)
            result[i] = soa[pos + pitch*(icy::SimParams::posx+i)];
    }
    else
    {
        for(int i=0; i<icy::SimParams::dim;i++)
            result[i] = data[icy::SimParams::posx+i];
    }
    return result;
}

double ProxyPoint::getValue(size_t valueIdx) const
{
    if(isReference)
        return soa[pos + pitch*valueIdx];
    else
        return data[valueIdx];
}


void ProxyPoint::setValue(size_t valueIdx, double value)
{
    if(isReference)
        soa[pos + pitch*valueIdx] = value;
    else
        data[valueIdx] = value;
}



bool ProxyPoint::getCrushedStatus()
{
    double dval;
    if(isReference)
    {
        dval = soa[pos + pitch*icy::SimParams::idx_utility_data];
    }
    else
    {
        dval = data[icy::SimParams::idx_utility_data];
    }
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0x10000);
}

bool ProxyPoint::getDisabledStatus()
{
    double dval;
    if(isReference)
    {
        dval = soa[pos + pitch*icy::SimParams::idx_utility_data];
    }
    else
    {
        dval = data[icy::SimParams::idx_utility_data];
    }
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0x20000);
}

uint16_t ProxyPoint::getGrain()
{
    double dval;
    if(isReference)
    {
        dval = soa[pos + pitch*icy::SimParams::idx_utility_data];
    }
    else
    {
        dval = data[icy::SimParams::idx_utility_data];
    }
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0xffff);
}



int ProxyPoint::getCellIndex(double hinv, unsigned GridY)
{
    Eigen::Vector2d v = getPos();
    Eigen::Vector2i idx = (v*hinv + Eigen::Vector2d::Constant(0.5)).cast<int>();
    return idx[0]*GridY + idx[1];
}

int ProxyPoint::getXIndex(double hinv) const
{
    double x = getValue(icy::SimParams::posx);
    int x_idx = (int)(x*hinv + 0.5);
    return x_idx;
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

