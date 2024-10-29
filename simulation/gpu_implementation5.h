#ifndef GPU_IMPLEMENTATION5_H
#define GPU_IMPLEMENTATION5_H


#include "gpu_partition.h"
#include "parameters_sim.h"
#include "point.h"
#include "host_side_soa.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <cuda.h>
#include <cuda_runtime.h>

#include <functional>


namespace icy { class Model; }


// contains information relevant to an individual data partition (which corresponds to a GPU device in multi-GPU setup)


class GPU_Implementation5
{
public:
    icy::Model *model;
    std::vector<GPU_Partition> partitions;
    HostSideSOA hssoa;
    Eigen::Vector2d indenter_force;

    std::function<void()> transfer_completion_callback;

    void allocate_arrays();
    void split_hssoa_into_partitions();     // perform grid and point partitioning
    void transfer_ponts_to_device();

    void initialize_and_enable_peer_access();
    void transfer_from_device();

    void synchronize(); // call before terminating the main thread
    void update_constants();
    void reset_grid();
    void reset_indenter_force_accumulator();

    void p2g();
    void receive_halos();
    void update_nodes(double simulation_time);
    void g2p(const bool recordPQ, const bool enablePointTransfer);
    void receive_points();
    void record_timings(const bool enablePointTransfer);

    // the size of this buffer (in the number of points) is stored in PointsHostBufferCapacity

private:

//    static void CUDART_CB callback_from_stream(cudaStream_t stream, cudaError_t status, void *userData);
};

#endif
