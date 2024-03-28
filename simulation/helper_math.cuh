#ifndef HELPER_MATH_CUH
#define HELPER_MATH_CUH

#include "givens.cuh"
#include <Eigen/Core>
#include <cuda.h>
#include <cuda_runtime.h>

#include "parameters_sim.h"
#include "point.h"




/**
 \brief 2x2 polar decomposition.
 \param[in] A matrix.
 \param[out] R Robustly a rotation matrix in givens form
 \param[out] S_Sym Symmetric. Whole matrix is stored

 Whole matrix S is stored since its faster to calculate due to simd vectorization
 Polar guarantees negative sign is on the small magnitude singular value.
 S is guaranteed to be the closest one to identity.
 R is guaranteed to be the closest rotation to A.
 */
template<typename T>
__device__ void polar_decomposition(const T a[4],
                GivensRotation<T>& r,
                T s[4]) {
    double x[2]		   = {a[0] + a[3], a[1] - a[2]};
    double denominator = sqrt(x[0] * x[0] + x[1] * x[1]);
    r.c				   = (T) 1;
    r.s				   = (T) 0;
    if(denominator != 0) {
        /*
      No need to use a tolerance here because x(0) and x(1) always have
      smaller magnitude then denominator, therefore overflow never happens.
    */
        r.c = x[0] / denominator;
        r.s = -x[1] / denominator;
    }
    for(int i = 0; i < 4; ++i) {
        s[i] = a[i];
    }
    r.template mat_rotation<2, T>(s);
}

/**
\brief 2x2 polar decomposition.
\param[in] A matrix.
\param[out] R Robustly a rotation matrix.
\param[out] S_Sym Symmetric. Whole matrix is stored

Whole matrix S is stored since its faster to calculate due to simd vectorization
Polar guarantees negative sign is on the small magnitude singular value.
S is guaranteed to be the closest one to identity.
R is guaranteed to be the closest rotation to A.
*/
template<typename T>
__device__ void polar_decomposition(const T a[4], T r[4], T s[4]) {
    GivensRotation<T> rotation(0, 1);
    polar_decomposition(a, rotation, s);
    rotation.fill<2>(r);
}


template <typename T> __device__ void inline my_swap(T& a, T& b)
{
    T c(a); a=b; b=c;
}


/**
\brief 2x2 SVD (singular value decomposition) A=USV'
\param[in] A Input matrix.
\param[out] u Robustly a rotation matrix in Givens form
\param[out] sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
\param[out] V Robustly a rotation matrix in Givens form
*/
template<typename T>
__forceinline__ __device__ void singular_value_decomposition(
        const T aa[4],
        GivensRotation<double>& u,
        T sigma[2],
        GivensRotation<double>& v) {

    double s_sym[4];///< column-major
    double a[4] {aa[0], aa[2], aa[1], aa[3]};
    polar_decomposition(a, u, s_sym);
    double cosine;
    double sine;
    double x  = s_sym[0];
    double y  = s_sym[2];
    double z  = s_sym[3];
    double y2 = y * y;
    if(y2 == 0) {
        // S is already diagonal
        cosine	 = 1;
        sine	 = 0;
        sigma[0] = x;
        sigma[1] = z;
    } else {
        double tau = T(0.5) * (x - z);
        double w   = sqrt(tau * tau + y2);
        // w > y > 0
        double t;
        if(tau > 0) {
            // tau + w > w > y > 0 ==> division is safe
            t = y / (tau + w);
        } else {
            // tau - w < -w < -y < 0 ==> division is safe
            t = y / (tau - w);
        }
        cosine = T(1) / sqrt(t * t + T(1));
        sine   = -t * cosine;
        /*
      v = [cosine -sine; sine cosine]
      sigma = v'SV. Only compute the diagonals for efficiency.
      Also utilize symmetry of S and don't form v yet.
    */
        double c2  = cosine * cosine;
        double csy = 2 * cosine * sine * y;
        double s2  = sine * sine;
        sigma[0]   = c2 * x - csy + s2 * z;
        sigma[1]   = s2 * x + csy + c2 * z;
    }

    // Sorting
    // Polar already guarantees negative sign is on the small magnitude singular value.
    if(sigma[0] < sigma[1])
    {
        my_swap(sigma[0], sigma[1]);
        v.c = -sine;
        v.s = cosine;
    } else {
        v.c = cosine;
        v.s = sine;
    }
    u *= v;
}



#endif // HELPER_MATH_CUH
