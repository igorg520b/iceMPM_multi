#include "host_side_soa.h"


void HostSideSOA::RemoveDisabledAndSort(double hinv, unsigned GridY)
{
    unsigned size_before = size;
    SOAIterator it_result = std::remove_if(begin(), end(), [](ProxyPoint &p){return p.getDisabledStatus();});
    size = it_result.m_point.pos;
    const double &hinv =
    std::sort(begin(), end(),
              [&hinv,&GridY](ProxyPoint &p1, ProxyPoint &p2)
              {return p1.getCellIndex(hinv,GridY)<p2.getCellIndex(hinv,GridY);});
    spdlog::info("RemoveDisabledAndSort: {} removed", size_before-size);
}


void HostSideSOA::Allocate(unsigned capacity)
{
    cudaFreeHost(host_buffer);
    this->capacity = capacity;
    cudaError_t err = cudaMallocHost(&points_host_buffer, sizeof(double)*capacity*icy::SimParams::nPtsArrays);
    if(err!=cudaSuccess) throw std::runtime_error("allocating host buffer for points");
    size = 0;
}



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

Eigen::Vector2d ProxyPoint::getPos()
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
    return (val & 0x1);
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
    return (val & 0x2);
}

uint8_t ProxyPoint::getGrain()
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
    return (val>>8) & 0xff;
}

double ProxyPoint::getJp_inv()
{
    double result;
    if(isReference)
    {
        result = soa[pos + pitch*icy::SimParams::idx_Jp_inv];
    }
    else
    {
        result = data[icy::SimParams::idx_Jp_inv];
    }
    return result;
}

std::pair<double,double> ProxyPoint::getPQ()
{
    double p,q;
    if(isReference)
    {
        p = soa[pos + pitch*icy::SimParams::idx_P];
        q = soa[pos + pitch*icy::SimParams::idx_Q];
    }
    else
    {
        p = data[icy::SimParams::idx_P];
        q = data[icy::SimParams::idx_Q];
    }
    return {p,q};
}


int ProxyPoint::getCellIndex(double hinv, unsigned GridY)
{
    Eigen::Vector2d v = getPos();
    Eigen::Vector2i idx = (v*hinv + Eigen::Vector2d::Constant(0.5)).cast<int>();
    return idx[0]*GridY + idx[1];
}

int ProxyPoint::getXIndex(double hinv)
{
    Eigen::Vector2d v = getPos();
    int x_idx = (int)(v[0]*hinv + 0.5);
    return x_idx;
}


// ==================================================== SOAIterator

SOAIterator::SOAIterator(unsigned pos, float *soa_data, unsigned pitch)
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
