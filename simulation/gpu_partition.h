#ifndef GPU_PARTITION_H
#define GPU_PARTITION_H

#include <Eigen/Core>
#include <Eigen/LU>
#include <spdlog/spdlog.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <functional>
#include "parameters_sim.h"
#include "point.h"



__global__ void v2_kernel_p2g();
__global__ void v2_kernel_g2p(bool recordPQ);
__global__ void v2_kernel_update_nodes(double indenter_x, double indenter_y);

__device__ Eigen::Matrix2d polar_decomp_R(const Eigen::Matrix2d &val);
__device__ void svd(const double a[4], double u[4], double sigma[2], double v[4]);
__device__ void svd2x2_modified(const Eigen::Matrix2d &mA, Eigen::Matrix2d &mU, Eigen::Vector2d &mS, Eigen::Matrix2d &mV);

__device__ void Wolper_Drucker_Prager(icy::Point &p);
__device__ void CheckIfPointIsInsideFailureSurface(icy::Point &p);
__device__ Eigen::Matrix2d KirchhoffStress_Wolper(const Eigen::Matrix2d &F);

__device__ void ComputePQ(icy::Point &p, const double &kappa, const double &mu);
__device__ void GetParametersForGrain(short grain, double &pmin, double &pmax, double &qmax, double &beta, double &mSq);

__device__ Eigen::Vector2d dev_d(Eigen::Vector2d Adiag);
__device__ Eigen::Matrix2d dev(Eigen::Matrix2d A);


struct GPU_Partition
{
    GPU_Partition();
    ~GPU_Partition();

    void initialize(int device, int partition);
    void update_constants();
    void allocate(unsigned n_points_capacity, unsigned grid_x_capacity);

    // host-side data
    int PartitionID;
    int Device;
    static icy::SimParams *prms;

    size_t nPtsPitch, nGridPitch; // in number of elements(!), for coalesced access on the device
    unsigned nPts_partition;    // actual number of points
    unsigned GridX_partition;   // size of the portion of the grid for which this partition is "responsible"
    unsigned GridX_offset;      // index where the grid starts in this partition
    size_t VectorCapacity_transfer;   // vector capacity for points that fly to another partition
    size_t VectorCapacity_disabled;   // for "disabled" points (points from the middle of the list that flew away
    double *host_side_indenter_force_accumulator;

    // stream and events
    cudaStream_t streamCompute;
    cudaEvent_t eventCycleStart, eventCycleStop;
    cudaEvent_t event_grid_halo_sent[2]; // 0-left; 1-right
    cudaEvent_t event_pts_sent[2]; // 0-left; 1-right

    bool initialized = false;
    uint8_t error_code = 0;

    // for tesitng and debugging
    unsigned transferred_point_count[2]; // left and right

    // device-side data
    double *pts_array, *grid_array, *indenter_force_accumulator;

    // Four GPU-side vectors to keep track of points that escape and arrive
    unsigned *_vector_data_disabled_points;  // list of indices <nPts_partition of "disabled" points
    // points that fly to/from the adjacent partitions (left-out, right-out, left-in, right-in)
    double *point_transfer_buffer[4];

/*
    __device__ static void insert_into_stack(unsigned *__data, unsigned value, uint8_t *_partition_error_indicator)
    {
        // data[0] is the current counter
        // data[1] is the capacity
        unsigned *counter = __data;
        unsigned *capacity = &__data[1];
        unsigned *index_list = &data[2];
        unsigned int idx = atomicAdd(counter, 1);
        if(idx < *capacity)
            index_list[idx] = value;
        else
            *_partition_error_indicator = 1;
    }
*/

};


#endif // GPU_PARTITION_H
