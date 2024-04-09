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

    int error_code;
    std::function<void()> transfer_completion_callback;

    void device_allocate_arrays();
    void transfer_ponts_to_device();

    void synchronize(); // call before terminating the main thread
    void update_constants();
    void reset_grid();
    void reset_indenter_force_accumulator();

    void p2g();
    void receive_halos();
    void update_nodes();
    void g2p(bool recordPQ);
    void receive_points();

    void cuda_transfer_from_device();

    // the size of this buffer (in the number of points) is stored in PointsHostBufferCapacity

private:

    static void CUDART_CB callback_from_stream(cudaStream_t stream, cudaError_t status, void *userData);
};

#endif
