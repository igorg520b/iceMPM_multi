#ifndef PARTICLE_H
#define PARTICLE_H

#include <cmath>
#include <utility>
#include <Eigen/Core>

#include "parameters_sim.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace icy { struct Point; }

struct icy::Point
{
    Eigen::Vector2d pos, velocity;
    Eigen::Matrix2d Bp, Fe; // refer to "The Material Point Method for Simulating Continuum Materials"

    double Jp_inv; // track the change in det(Fp)
    short grain;

    double p_tr, q_tr, Je_tr;
    Eigen::Matrix2d U, V;
    Eigen::Vector2d vSigma, vSigmaSquared, v_s_hat_tr;

    long long utility_data;
};


#endif // PARTICLE_H
