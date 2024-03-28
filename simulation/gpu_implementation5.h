#ifndef GPU_IMPLEMENTATION5_H
#define GPU_IMPLEMENTATION5_H


#include "gpu_partition.h"
#include "parameters_sim.h"
#include "point.h"

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
    int error_code;
    std::function<void()> transfer_completion_callback;

    void initialize();
    void test();
    void synchronize(); // call before terminating the main thread
    void cuda_update_constants();
    void cuda_allocate_arrays(size_t nGridNodes, size_t nPoints);
    void cuda_reset_grid();
    void transfer_ponts_to_device();
    void cuda_p2g();
    void cuda_g2p(bool recordPQ);
    void cuda_update_nodes(double indenter_x, double indenter_y);
    void cuda_reset_indenter_force_accumulator();

    void cuda_transfer_from_device();

    cudaEvent_t eventCycleStart, eventCycleStop;

    double *tmp_transfer_buffer = nullptr; // buffer in page-locked memory for transferring the data between device and host
    double *host_side_indenter_force_accumulator = nullptr;

private:


    static void CUDART_CB callback_from_stream(cudaStream_t stream, cudaError_t status, void *userData);
};

#endif
