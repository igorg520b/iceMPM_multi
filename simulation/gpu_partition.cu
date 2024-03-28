#include "gpu_partition.h"

__device__ uint8_t gpu_error_indicator;
__constant__ icy::SimParams gprms;

icy::SimParams *GPU_Partition::prms;

GPU_Partition::GPU_Partition(int device, int partition)
{
    this->PartitionID = partition;
    this->Device = device;
    host_side_indenter_force_accumulator = nullptr;
    pts_array = nullptr;
    grid_array = nullptr;
    indenter_force_accumulator = nullptr;
    _vector_data_disabled_points = nullptr;
    for(int i=0;i<4;i++) point_transfer_buffer[i] = nullptr;
}


void GPU_Partition::initialize(int device, int partition)
{
    cudaSetDevice(Device);
    cudaEventCreate(&eventCycleStart);
    cudaEventCreate(&eventCycleStop);
    cudaEventCreate(&event_grid_halo_sent[0]);
    cudaEventCreate(&event_grid_halo_sent[1]);
    cudaEventCreate(&event_pts_sent[0]);
    cudaEventCreate(&event_pts_sent[1]);
    cudaError_t err = cudaStreamCreate(&streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition initialization failure");
    initialized = true;
    spdlog::info("Initialized partition {} on device {}", partition, device);
}


void GPU_Partition::update_constants()
{
    cudaSetDevice(Device);
    cudaError_t err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(error_code));
    if(err != cudaSuccess) throw std::runtime_error("gpu_error_indicator initialization");
    err = cudaMemcpyToSymbol(gprms, prms, sizeof(icy::SimParams));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: gprms");
    spdlog::info("Constant symbols copied to device {}; partition {}", Device, PartitionID);
}

void GPU_Partition::allocate(unsigned n_points, unsigned gridx_partition, unsigned grid_x_offset)
{
    cudaSetDevice(Device);

    // host-side indenter accumulator
    cudaFreeHost(host_side_indenter_force_accumulator);
    cudaError_t err = cudaMallocHost(&host_side_indenter_force_accumulator, prms->IndenterArraySize());
    if(err!=cudaSuccess) throw std::runtime_error("GPU_Partition allocate host-side buffer");

    // indenter accumulator
    cudaFree(indenter_force_accumulator);
    err = cudaMalloc(&indenter_force_accumulator, prms->IndenterArraySize());
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    // points
    cudaFree(pts_array);
    const size_t pts_buffer_requested = sizeof(double) * n_points * (1 + prms->ExtraSpaceForIncomingPoints);
    err = cudaMallocPitch(&pts_array, &nPtsPitch, pts_buffer_requested, icy::SimParams::nPtsArrays);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate");
    nPtsPitch /= sizeof(double);

    // point transfer buffers
    for(int i=0;i<4;i++) cudaFree(point_transfer_buffer[i]);
    VectorCapacity_transfer = icy::SimParams::nPtsArrays * n_points * prms->PointsTransferBufferFraction;
    for(int i=0;i<4;i++)
    {
        err = cudaMalloc(&point_transfer_buffer[i], VectorCapacity_transfer*sizeof(double));
        if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate");
    }

    // integer vector for disabled points
    cudaFree(_vector_data_disabled_points);
    VectorCapacity_disabled = npts_wanted * prms->ExtraSpaceForIncomingPoints;
    err = cudaMalloc(&point_transfer_buffer[i], VectorCapacity_disabled*sizeof(unsigned));
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate");

    // grid
    cudaFree(grid_array);
    GridX_partition = gridx_partition;
    GridX_offset = grid_x_offset;
    size_t GridX_total_local = GridX_partition + 4*prms->GridHaloSize; // x2 halos on left and right)
    size_t grid_size_local = prms->GridY * GridX_total_local;
    err = cudaMallocPitch (&grid_array, &nGridPitch, sizeof(double)*grid_size_local, icy::SimParams::nGridArrays);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate grid array");
    nGridPitch /= sizeof(double); // assume that this divides without remainder

    spdlog::info("Allocate done on partition {}; device {}", PartitionID, Device);
}
