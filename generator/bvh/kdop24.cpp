#include "kdop24.h"
#include <cfloat>


void kDOP24::Reset()
{
    for(int i=0;i<12;i++)
    {
        d[i]    =  FLT_MAX;
        d[i+12] = -FLT_MAX;
    }
}

bool kDOP24::Overlaps(kDOP24 &b)
{
    for(int i=0;i<12;i++)
    {
        if (d[i] > b.d[i+12]) return false;
        if (d[i+12] < b.d[i]) return false;
    }
    return true;
}

void kDOP24::Expand(float x, float y, float z)
{
    MinMax(x, d[0], d[12]);
    MinMax(y, d[1], d[13]);
    MinMax(z, d[2], d[14]);

    MinMax(x+y, d[3], d[15]);
    MinMax(x+z, d[4], d[16]);
    MinMax(y+z, d[5], d[17]);
    MinMax(x-y, d[6], d[18]);
    MinMax(x-z, d[7], d[19]);
    MinMax(y-z, d[8], d[20]);
    MinMax(x+y-z, d[9], d[21]);
    MinMax(x+z-y, d[10], d[22]);
    MinMax(y+z-x, d[11], d[23]);

    ctrX = (d[0]+d[12])/2;
    ctrY = (d[1]+d[13])/2;
    ctrZ = (d[2]+d[14])/2;

    dX = d[12]-d[0];
    dY = d[13]-d[1];
    dZ = d[14]-d[2];
}

void kDOP24::Expand(kDOP24 &b)
{
    for(int i=0;i<12;i++)
    {
        if(d[i] > b.d[i]) d[i]=b.d[i];
        if(d[i+12] < b.d[i+12]) d[i+12] = b.d[i+12];
    }

    ctrX = (d[0]+d[12])/2;
    ctrY = (d[1]+d[13])/2;
    ctrZ = (d[2]+d[14])/2;

    dX = d[12]-d[0];
    dY = d[13]-d[1];
    dZ = d[14]-d[2];
}


void kDOP24::MinMax(float p, float &mi, float &ma)
{
    if (p > ma) ma = p;
    if (p < mi) mi = p;
}

