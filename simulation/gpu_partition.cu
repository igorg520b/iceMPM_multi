#include "gpu_partition.h"

constexpr double d = 2; // dimensions
constexpr double coeff1 = 1.4142135623730950; // sqrt((6-d)/2.);

__device__ uint8_t gpu_error_indicator;
__constant__ icy::SimParams gprms;

icy::SimParams *GPU_Partition::prms;

void GPU_Partition::transfer_points_from_soa_to_device(HostSideSOA &hssoa, unsigned point_idx_offset)
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(err != cudaSuccess)
    {
        spdlog::critical("error setting the device {} in transfer points",Device);
        throw std::runtime_error("transfer_points_from_soa_to_device");
    }

    // due to the layout of host-side SOA, we transfer the pts arrays one-by-one
    for(int i=0;i<icy::SimParams::nPtsArrays;i++)
    {
        double *ptr_dst = pts_array + i*nPtsPitch;
        double *ptr_src = hssoa.getPointerToLine(i)+point_idx_offset;
        err = cudaMemcpyAsync(ptr_dst, ptr_src, nPts_partition*sizeof(double), cudaMemcpyHostToDevice, streamCompute);
        if(err != cudaSuccess)
        {
            const char* errorString = cudaGetErrorString(err);
            spdlog::critical("PID {}; line {}; nPts_partition {}, cuda error: {}",PartitionID, i, nPts_partition, errorString);
            throw std::runtime_error("transfer_points_from_soa_to_device");
        }
    }
}


GPU_Partition::GPU_Partition()
{
    nPts_partition = GridX_partition = GridX_offset = 0;
    host_side_indenter_force_accumulator = nullptr;
    pts_array = nullptr;
    grid_array = nullptr;
    indenter_force_accumulator = nullptr;
    _vector_data_disabled_points = nullptr;
    for(int i=0;i<4;i++) point_transfer_buffer[i] = nullptr;
}

GPU_Partition::~GPU_Partition()
{
    cudaSetDevice(Device);
    cudaEventDestroy(eventCycleStart);
    cudaEventDestroy(eventCycleStop);
    cudaEventDestroy(event_grid_halo_sent[0]);
    cudaEventDestroy(event_grid_halo_sent[1]);
    cudaEventDestroy(event_pts_sent[0]);
    cudaEventDestroy(event_pts_sent[1]);

    cudaStreamDestroy(streamCompute);

    cudaFreeHost(host_side_indenter_force_accumulator);
    cudaFree(indenter_force_accumulator);
    cudaFree(pts_array);
    for(int i=0;i<4;i++) cudaFree(point_transfer_buffer[i]);
    cudaFree(_vector_data_disabled_points);
    cudaFree(grid_array);
    spdlog::info("Destructor invoked; partition {} on device {}", PartitionID, Device);
}

void GPU_Partition::initialize(int device, int partition)
{
    this->PartitionID = partition;
    this->Device = device;
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

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);
    spdlog::info("Partition {}: initialized dev {}; compute {}.{}", PartitionID, Device,deviceProp.major, deviceProp.minor);
}


void GPU_Partition::allocate(unsigned n_points_capacity, unsigned grid_x_capacity)
{
    cudaSetDevice(Device);

    // host-side indenter accumulator
    cudaError_t err = cudaMallocHost(&host_side_indenter_force_accumulator, prms->IndenterArraySize());
    if(err!=cudaSuccess) throw std::runtime_error("GPU_Partition allocate host-side buffer");
    memset(host_side_indenter_force_accumulator, 0, prms->IndenterArraySize());

    // indenter accumulator
    err = cudaMalloc(&indenter_force_accumulator, prms->IndenterArraySize());
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    // points
    const size_t pts_buffer_requested = sizeof(double) * n_points_capacity;
    err = cudaMallocPitch(&pts_array, &nPtsPitch, pts_buffer_requested, icy::SimParams::nPtsArrays);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate");
    nPtsPitch /= sizeof(double);

    // point transfer buffers
    VectorCapacity_transfer = n_points_capacity * prms->PointsTransferBufferFraction;
    for(int i=0;i<4;i++)
    {
        err = cudaMalloc(&point_transfer_buffer[i], (1+VectorCapacity_transfer*icy::SimParams::nPtsArrays)*sizeof(double));
        if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate");
    }

    // integer vector for disabled points
    VectorCapacity_disabled = n_points_capacity * prms->ExtraSpaceForIncomingPoints;
    err = cudaMalloc(&_vector_data_disabled_points, (VectorCapacity_disabled+1)*sizeof(unsigned));
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate _vector_data_disabled_points");

    // grid
    size_t grid_size_local_requested = prms->GridY*(grid_x_capacity + 4*prms->GridHaloSize) * sizeof(double);
    err = cudaMallocPitch (&grid_array, &nGridPitch, grid_size_local_requested, icy::SimParams::nGridArrays);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate grid array");
    nGridPitch /= sizeof(double); // assume that this divides without remainder

    spdlog::info("Partition {}-{}: allocated GridPitch {} ({}); Pts {}; Disabled {}; PtsTransfer {}; grid_size_local_requested {}",
                 PartitionID, Device, nGridPitch, nGridPitch/prms->GridY, nPtsPitch, VectorCapacity_disabled, VectorCapacity_transfer, grid_size_local_requested);
}


void GPU_Partition::clear_utility_vectors()
{
    spdlog::info("P {} D {}, utility vectors clear",PartitionID,Device);
    cudaSetDevice(Device);
    cudaError_t err = cudaMemsetAsync(_vector_data_disabled_points, 0, sizeof(unsigned), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("initialize_utility_vectors");
    for(int i=0;i<4;i++)
    {
        cudaError_t err = cudaMemsetAsync(point_transfer_buffer[i], 0, sizeof(double), streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("initialize_utility_vectors");
    }
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


void GPU_Partition::reset_grid()
{
    cudaSetDevice(Device);

    size_t gridArraySize = nGridPitch * icy::SimParams::nGridArrays * sizeof(double);
    cudaError_t err = cudaMemsetAsync(grid_array, 0, gridArraySize, streamCompute);
    if(err != cudaSuccess)
    {
        const char* errorString = cudaGetErrorString(err);
        spdlog::critical("P {}; cuda_reset_grid error: {}",PartitionID, errorString);
        spdlog::critical("nGridPitch {}; GridY {}; gridArraySize {}", nGridPitch, prms->GridY, gridArraySize);
        throw std::runtime_error("cuda_reset_grid error");
    }
}


void GPU_Partition::reset_indenter_force_accumulator()
{
    cudaSetDevice(Device);
    cudaError_t err = cudaMemsetAsync(indenter_force_accumulator, 0, prms->IndenterArraySize(), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}
