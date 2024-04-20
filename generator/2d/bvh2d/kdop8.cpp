#include "kdop8.h"
#include <cfloat>
#include <iostream>

void kDOP8::Reset()
{
    d[0] = d[1] = d[2] = d[3] = DBL_MAX;
    g[0] = g[1] = g[2] = g[3] = -DBL_MAX;
}

bool kDOP8::Overlaps(kDOP8 &b)
{
    if (d[0] > b.g[0]) return false;
    if (d[1] > b.g[1]) return false;
    if (d[2] > b.g[2]) return false;
    if (d[3] > b.g[3]) return false;

    if (g[0] < b.d[0]) return false;
    if (g[1] < b.d[1]) return false;
    if (g[2] < b.d[2]) return false;
    if (g[3] < b.d[3]) return false;

    return true;
}

void kDOP8::Expand(float x, float y)
{
    MinMax(x, d[0], g[0]);
    MinMax(y, d[1], g[1]);
    MinMax(x+y, d[2], g[2]);
    MinMax(x-y, d[3], g[3]);

    ctrX = (d[0] + g[0]) / 2;
    ctrY = (d[1] + g[1]) / 2;
    dX = g[0] - d[0];
    dY = g[1] - d[1];
}

void kDOP8::Expand(kDOP8 &b)
{
    for(int i=0;i<4;i++)
    {
        if(d[i]>b.d[i]) d[i]=b.d[i];
        if(g[i]<b.g[i]) g[i]=b.g[i];
    }
    ctrX = (d[0] + g[0]) / 2;
    ctrY = (d[1] + g[1]) / 2;
    dX = g[0] - d[0];
    dY = g[1] - d[1];
}

void kDOP8::ExpandBy(float radius)
{
    for(unsigned i=0;i<4;i++)
    {
        d[i]-=radius;
        g[i]+=radius;
    }
}

void kDOP8::MinMax(float p, float &mi, float &ma)
{
    if (p < mi) mi = p;
    if (p > ma) ma = p;
}
