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
#include "host_side_soa.h"


// kernels
__global__ void partition_kernel_p2g(const int gridX, const int gridX_offset, const int pitch_grid,
                              const int count_pts, const int pitch_pts,
                                     const double *buffer_pts, double *buffer_grid);


// receive grid data from adjacent partitions
__global__ void partition_kernel_receive_halos(const int haloElementCount, const int gridX,
                                               const int pitch_grid, double *buffer_grid);


__global__ void partition_kernel_update_nodes(const Eigen::Vector2d indCenter,
                                              const int nNodes, const int gridX_offset, const int pitch_grid,
                                              double *_buffer_grid, double *indenter_force_accumulator);


__global__ void partition_kernel_g2p(const bool recordPQ,
                                     const int gridX, const int gridX_offset, const int pitch_grid,
                                     const int count_pts, const int pitch_pts,
                                     double *buffer_pts, const double *buffer_grid,
                                     double *_point_transfer_buffer[4],
                                     int *utility_data,
                                     const int VectorCapacity_transfer, const int VectorCapacity_disabled);


// take points from the receive buffer and add them to the list
__global__ void partition_kernel_receive_points(const int count_left, const int count_right,
                                                const int count_pts, const int pitch_pts,
                                                double *buffer_pts,
                                                double *point_transfer_buffer[4],
                                                int *utility_data);

// device functions used by kernels
__device__ void PreparePointForTransfer(const int pt_idx, const int whichSide, double *_point_transfer_buffer[4],
                                        int *vector_data_disabled_points, int *utility_data,
                                        icy::Point &p,
                                        const int VectorCapacity_transfer, const int VectorCapacity_disabled);

__device__ Eigen::Matrix2d polar_decomp_R(const Eigen::Matrix2d &val);
__device__ void svd(const double a[4], double u[4], double sigma[2], double v[4]);
__device__ void svd2x2(const Eigen::Matrix2d &mA, Eigen::Matrix2d &mU, Eigen::Vector2d &mS, Eigen::Matrix2d &mV);

__device__ void Wolper_Drucker_Prager(icy::Point &p);
__device__ void CheckIfPointIsInsideFailureSurface(icy::Point &p);
__device__ Eigen::Matrix2d KirchhoffStress_Wolper(const Eigen::Matrix2d &F);

__device__ void ComputePQ(icy::Point &p, const double &kappa, const double &mu);
__device__ void GetParametersForGrain(short grain, double &pmin, double &pmax, double &qmax, double &beta, double &mSq);

__device__ Eigen::Vector2d dev_d(Eigen::Vector2d Adiag);
__device__ Eigen::Matrix2d dev(Eigen::Matrix2d A);


struct GPU_Partition
{
    constexpr static int idx_transfer_to_left = 0;
    constexpr static int idx_transfer_to_right = 1;
    constexpr static int idx_points_added_to_soa = 2;
    constexpr static int idx_disabled_indices = 3;
    constexpr static size_t utility_data_size = 4;

    GPU_Partition();
    ~GPU_Partition();

    // preparation
    void initialize(int device, int partition);
    void allocate(int n_points_capacity, int grid_x_capacity);
    void transfer_points_from_soa_to_device(HostSideSOA &hssoa, int point_idx_offset);
    void clear_utility_vectors();
    void update_constants();
    void transfer_from_device(HostSideSOA &hssoa, int point_idx_offset);

    // calculation
    void reset_grid();
    void reset_indenter_force_accumulator();
    void p2g();
    void receive_halos();   // neightbour halos were copied, but we need to incorporate them into the grid
    void update_nodes();
    void g2p(const bool recordPQ);
    void receive_points(int nFromLeft, int nFromRight);

    // helper functions
    double *getHaloAddress(int whichHalo, int whichGridArray);
    double *getHaloReceiveAddress(int whichHalo, int whichGridArray);
    int getLeftBufferCount() {return host_side_utility_data[idx_transfer_to_left];}
    int getRightBufferCount() {return host_side_utility_data[idx_transfer_to_right];}
    int getAddedPtsCount() {return host_side_utility_data[idx_points_added_to_soa];}
    int getDisabledPtsCount() {return host_side_utility_data[idx_disabled_indices];}

    // host-side data
    int PartitionID;
    int Device;
    static icy::SimParams *prms;

    size_t nPtsPitch, nGridPitch; // in number of elements(!), for coalesced access on the device
    int nPts_partition;    // actual number of points (including disabled)
    int GridX_partition;   // size of the portion of the grid for which this partition is "responsible"
    int GridX_offset;      // index where the grid starts in this partition

    double *host_side_indenter_force_accumulator;
    int *host_side_utility_data; // sizes of outbound pt tranfer buffers (2), disabled pts (1)

    // stream and events
    cudaStream_t streamCompute;
    cudaEvent_t eventCycleStart, eventCycleStop;
    cudaEvent_t event_grid_halo_sent;
    cudaEvent_t event_pts_sent;
    cudaEvent_t event_utility_data_transferred;

    bool initialized = false;
    uint8_t error_code = 0;


    // device-side data
    int *device_side_utility_data;
    double *pts_array, *grid_array, *indenter_force_accumulator;

    // Four GPU-side vectors to keep track of points that escape and arrive
//    int *vector_data_disabled_points;  // list of indices <nPts_partition of "disabled" points
    // points that fly to/from the adjacent partitions (left-out, right-out, left-in, right-in)
    double *point_transfer_buffer[4];
};


#endif // GPU_PARTITION_H
