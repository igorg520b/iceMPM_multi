#include "proxypoint2d.h"

// ====================================================== ProxyPoint
ProxyPoint::ProxyPoint(const ProxyPoint &other)
{
    isReference = false;
    *this = other;
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

Eigen::Vector2d ProxyPoint::getVelocity() const
{
    Eigen::Vector2d result;
    if(isReference)
    {
        for(int i=0; i<icy::SimParams::dim;i++)
            result[i] = soa[pos + pitch*(icy::SimParams::velx+i)];
    }
    else
    {
        for(int i=0; i<icy::SimParams::dim;i++)
            result[i] = data[icy::SimParams::velx+i];
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


void ProxyPoint::setPartition(uint8_t PartitionID)
{
    // retrieve the existing value
    double dval = getValue(icy::SimParams::idx_utility_data);
    long long val = *reinterpret_cast<long long*>(&dval);

    long long _pid = (long long)PartitionID;
    _pid <<= 24;
    val &= 0xffffffff00ffffffll;
    val |= _pid;

    long long *ptr;
    if(isReference)
    {
        ptr = (long long*)&soa[pos + pitch*icy::SimParams::idx_utility_data];
    }
    else
    {
        ptr = (long long*)&data[icy::SimParams::idx_utility_data];
    }
    *ptr = val;
}


uint8_t ProxyPoint::getPartition()
{
    double dval = getValue(icy::SimParams::idx_utility_data);
    long long val = *reinterpret_cast<long long*>(&dval);
    val >>= 24;
    return (uint8_t)(val & 0xff);
}



bool ProxyPoint::getCrushedStatus()
{
    double dval = getValue(icy::SimParams::idx_utility_data);
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0x10000);
}

bool ProxyPoint::getDisabledStatus()
{
    double dval = getValue(icy::SimParams::idx_utility_data);
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0x20000ll) == 0x20000ll;
}

bool ProxyPoint::getLiquidStatus()
{
    double dval = getValue(icy::SimParams::idx_utility_data);
    long long val = *reinterpret_cast<long long*>(&dval);
    return (val & 0x40000ll) == 0x40000ll;
}

void ProxyPoint::setLiquidStatus(bool setval)
{
    double dval = getValue(icy::SimParams::idx_utility_data);
    long long llval = *reinterpret_cast<long long*>(&dval);
    if(setval) llval |= 0x40000ll;
    else llval &= ~0x40000ll;
    setValue(icy::SimParams::idx_utility_data, *reinterpret_cast<double*>(&llval));
}


uint16_t ProxyPoint::getGrain()
{
    double dval = getValue(icy::SimParams::idx_utility_data);
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
