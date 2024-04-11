#include "point.h"

#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Core>


void icy::Point::Reset()
{
    Fe.setIdentity();
    velocity.setZero();
    Bp.setZero();
    Jp_inv = 1;
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

Eigen::Vector2d icy::Point::getPos(const double *buffer, const int pitch, const int point_index)
{
    Eigen::Vector2d result;
    for(int i=0; i<icy::SimParams::dim;i++) result[i] = buffer[point_index + pitch*(icy::SimParams::posx+i)];
    return result;
}


uint8_t icy::Point::getCrushedStatus(const double *buffer, const int pitch, const int point_index)
{
    uint8_t* ptr_intact = (uint8_t*)(&buffer[pitch*icy::SimParams::idx_utility_data]);
    return ptr_intact[point_index];
}

double icy::Point::getJp_inv(const double *buffer, const int pitch, const int point_index)
{
    return buffer[point_index + pitch*icy::SimParams::idx_Jp_inv];
}

std::pair<double,double> icy::Point::getPQ(const double *buffer, const int pitch, const int point_index)
{
    return {buffer[point_index + pitch*icy::SimParams::idx_P],buffer[point_index + pitch*icy::SimParams::idx_Q]};
}


short icy::Point::getGrain(const double *buffer, const int pitch, const int point_index)
{
    char* ptr_intact = (char*)(&buffer[pitch*icy::SimParams::idx_utility_data]);
    short* ptr_grain = (short*)(&ptr_intact[pitch]);
    short grain = ptr_grain[point_index];
    return grain;
}
*/
