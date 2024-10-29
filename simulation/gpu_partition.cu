#include "gpu_partition.h"
#include "helper_math.cuh"
#include "kernels.cuh"
#include <stdio.h>



icy::SimParams *GPU_Partition::prms;



// =========================================  GPU_Partition class



void GPU_Partition::transfer_from_device(HostSideSOA &hssoa, int point_idx_offset)
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(err != cudaSuccess) throw std::runtime_error("transfer_from_device() set");

    for(int j=0;j<icy::SimParams::nPtsArrays;j++)
    {
        if((point_idx_offset + nPts_partition) > hssoa.capacity)
            throw std::runtime_error("transfer_from_device() HSSOA capacity");

        double *ptr_src = pts_array + j*nPtsPitch;
        double *ptr_dst = hssoa.getPointerToLine(j)+point_idx_offset;

        err = cudaMemcpyAsync(ptr_dst, ptr_src, nPts_partition*sizeof(double), cudaMemcpyDeviceToHost, streamCompute);
        if(err != cudaSuccess)
        {
            const char *errorString = cudaGetErrorString(err);
            spdlog::critical("error when copying points: {}", errorString);
            throw std::runtime_error("transfer_from_device() cudaMemcpyAsync points");
        }

        err = cudaMemcpyAsync(host_side_indenter_force_accumulator, indenter_force_accumulator,
                              prms->IndenterArraySize(), cudaMemcpyDeviceToHost, streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("transfer_from_device() cudaMemcpyAsync indenter");

        err = cudaMemcpyFromSymbolAsync(&error_code, gpu_error_indicator, sizeof(error_code), 0, cudaMemcpyDeviceToHost,
                                        streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("transfer_from_device");
    }
}


void GPU_Partition::transfer_points_from_soa_to_device(HostSideSOA &hssoa, int point_idx_offset)
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_from_soa_to_device");

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
    error_code = 0;
    nPts_partition = GridX_partition = GridX_offset = 0;
    nPts_disabled = 0;

    host_side_indenter_force_accumulator = nullptr;
    host_side_utility_data = nullptr;
    pts_array = nullptr;
    grid_array = nullptr;
    indenter_force_accumulator = nullptr;
    for(int i=0;i<4;i++) point_transfer_buffer[i] = nullptr;
    device_side_utility_data = nullptr;
    halo_transfer_buffer[0] = nullptr;
    halo_transfer_buffer[1] = nullptr;
}

GPU_Partition::~GPU_Partition()
{
    cudaSetDevice(Device);

    cudaEventDestroy(event_10_cycle_start);
    cudaEventDestroy(event_20_grid_halo_sent);
    cudaEventDestroy(event_30_halo_accepted);
    cudaEventDestroy(event_40_grid_updated);
    cudaEventDestroy(event_50_g2p_completed);
    cudaEventDestroy(event_70_pts_sent);
    cudaEventDestroy(event_80_pts_accepted);

    cudaStreamDestroy(streamCompute);

    cudaFreeHost(host_side_indenter_force_accumulator);
    cudaFreeHost(host_side_utility_data);

    cudaFree(grid_array);
    cudaFree(pts_array);
    cudaFree(indenter_force_accumulator);
    for(int i=0;i<4;i++) cudaFree(point_transfer_buffer[i]);
    cudaFree(device_side_utility_data);
    spdlog::info("Destructor invoked; partition {} on device {}", PartitionID, Device);
}

void GPU_Partition::initialize(int device, int partition)
{
    this->PartitionID = partition;
    this->Device = device;
    cudaSetDevice(Device);

    cudaEventCreate(&event_10_cycle_start);
    cudaEventCreate(&event_20_grid_halo_sent);
    cudaEventCreate(&event_30_halo_accepted);
    cudaEventCreate(&event_40_grid_updated);
    cudaEventCreate(&event_50_g2p_completed);
    cudaEventCreate(&event_70_pts_sent);
    cudaEventCreate(&event_80_pts_accepted);

    cudaError_t err = cudaStreamCreate(&streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition initialization failure");
    initialized = true;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);
    spdlog::info("Partition {}: initialized dev {}; compute {}.{}", PartitionID, Device,deviceProp.major, deviceProp.minor);
}


void GPU_Partition::allocate(int n_points_capacity, int gx)
{
    cudaError_t err;
    cudaSetDevice(Device);
    spdlog::info("P{}-{} allocate", PartitionID, Device);

    // grid
    const int &halo = prms->GridHaloSize;
    const int &gy = prms->GridY;

    size_t total_device = 0;
    size_t grid_size_local_requested = sizeof(double) * gy * (gx + 6*halo);
    err = cudaMallocPitch (&grid_array, &nGridPitch, grid_size_local_requested, icy::SimParams::nGridArrays);
    total_device += nGridPitch * icy::SimParams::nGridArrays;
    if(err != cudaSuccess)
    {
        const char *s = cudaGetErrorString(err);
        spdlog::error("err {}: {}", err, s);
        throw std::runtime_error("GPU_Partition allocate grid array");
    }
    nGridPitch /= sizeof(double); // assume that this divides without remainder

    halo_transfer_buffer[0] = grid_array + gy*(gx+2*halo);
    halo_transfer_buffer[1] = grid_array + gy*(gx+4*halo);

    // host-side indenter accumulator
    err = cudaMallocHost(&host_side_indenter_force_accumulator, prms->IndenterArraySize());
    if(err!=cudaSuccess) throw std::runtime_error("GPU_Partition allocate host-side buffer");
    memset(host_side_indenter_force_accumulator, 0, prms->IndenterArraySize());

    // indenter accumulator
    err = cudaMalloc(&indenter_force_accumulator, prms->IndenterArraySize());
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    // points
    const size_t pts_buffer_requested = sizeof(double) * n_points_capacity;
    err = cudaMallocPitch(&pts_array, &nPtsPitch, pts_buffer_requested, icy::SimParams::nPtsArrays);
    total_device += nPtsPitch * icy::SimParams::nPtsArrays;
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate");
    nPtsPitch /= sizeof(double);

    // point transfer buffers
    for(int i=0;i<4;i++)
    {
        size_t count = prms->VectorCapacity_transfer*icy::SimParams::nPtsArrays*sizeof(double);
        err = cudaMalloc(&point_transfer_buffer[i], count);
        total_device += count;
        if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate");
    }

    err = cudaMallocHost(&host_side_utility_data, utility_data_size*sizeof(int));
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate host-side array");

    err = cudaMalloc(&device_side_utility_data, utility_data_size*sizeof(int));
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate space for utility data");

    spdlog::info("allocate: P {}-{}:  GridPitch/Y {}; Pts {}; PtsTransfer {}; total {:.2} Mb",
                 PartitionID, Device,
                 nGridPitch/prms->GridY, nPtsPitch,
                 prms->VectorCapacity_transfer, (double)total_device/(1024*1024));
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



void GPU_Partition::reset_indenter_force_accumulator()
{
    cudaSetDevice(Device);
    cudaError_t err = cudaMemsetAsync(indenter_force_accumulator, 0, prms->IndenterArraySize(), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}


// ============================================================= main simulation steps
void GPU_Partition::reset_grid()
{
    cudaError_t err;
    cudaSetDevice(Device);
    size_t gridArraySize = nGridPitch * icy::SimParams::nGridArrays * sizeof(double);
    err = cudaMemsetAsync(grid_array, 0, gridArraySize, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}

void GPU_Partition::p2g()
{
    cudaSetDevice(Device);
    const int gridX = prms->GridXTotal; // todo: change to gridx_partition
    const int gridXoffset = GridX_offset;

    const int &n = nPts_partition;
    const int &tpb = prms->tpb_P2G;
    const int blocksPerGrid = (n + tpb - 1) / tpb;
    partition_kernel_p2g<<<blocksPerGrid, tpb, 0, streamCompute>>>(gridX, gridXoffset, nGridPitch,
                         nPts_partition, nPtsPitch, pts_array, grid_array);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("p2g kernel");
}

void GPU_Partition::update_nodes(double simulation_time)
{
    cudaSetDevice(Device);
    const int nGridNodes = prms->GridY * (GridX_partition + 2*prms->GridHaloSize);

    int tpb = prms->tpb_Upd;
    int nBlocks = (nGridNodes + tpb - 1) / tpb;
    Eigen::Vector2d ind_center(prms->indenter_x, prms->indenter_y);

    partition_kernel_update_nodes<<<nBlocks, tpb, 0, streamCompute>>>(ind_center, nGridNodes, GridX_offset,
                                                                      nGridPitch, grid_array, indenter_force_accumulator,
                                                                      simulation_time);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("update_nodes");
}

void GPU_Partition::g2p(const bool recordPQ, const bool enablePointTransfer)
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("g2p cudaSetDevice");

    if(enablePointTransfer)
    {
        // clear the counters for (0) left transfer, (1) right transfer, (2) added to the end of the list
        err = cudaMemsetAsync(device_side_utility_data, 0, utility_data_size*sizeof(int), streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("g2p cudaMemset");
    }

    const int &n = nPts_partition;
    const int &tpb = prms->tpb_G2P;
    const int nBlocks = (n + tpb - 1) / tpb;

    partition_kernel_g2p<<<nBlocks, tpb, 0, streamCompute>>>(recordPQ, enablePointTransfer,
                                                             GridX_partition, GridX_offset, nGridPitch,
                                                             nPts_partition, nPtsPitch,
                                                             pts_array, grid_array,
                                                             device_side_utility_data,
                                                             prms->VectorCapacity_transfer,
                                                             point_transfer_buffer[0],
                                                             point_transfer_buffer[1]);

    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("g2p kernel");

    if(enablePointTransfer)
    {
        err = cudaMemcpyAsync(host_side_utility_data, device_side_utility_data, sizeof(int)*utility_data_size,
                              cudaMemcpyDeviceToHost, streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("GPU_Partition::g2p cudaMemcpy");
    }
}


void GPU_Partition::receive_points(int nFromLeft, int nFromRight)
{
    if(nFromLeft)
    {
        const int &n = nFromLeft;
        const int &tpb = 64;
        const int nBlocks = (n + tpb - 1) / tpb;
        partition_kernel_receive_points<<<nBlocks, tpb, 0, streamCompute>>>(n,
                                                                            nPts_partition, nPtsPitch,
                                                                            pts_array,
                                                                            point_transfer_buffer[2]);
        if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("receive_nodes kernel execution left");
        nPts_partition += n;
    }
    if(nFromRight)
    {
        const int &n = nFromRight;
        const int &tpb = 64;
        const int nBlocks = (n + tpb - 1) / tpb;
        partition_kernel_receive_points<<<nBlocks, tpb, 0, streamCompute>>>(n,
                                                                            nPts_partition, nPtsPitch,
                                                                            pts_array,
                                                                            point_transfer_buffer[3]);
        if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("receive_nodes kernel execution right");
        nPts_partition += n;
    }
}

void GPU_Partition::receive_halos()
{
    cudaSetDevice(Device);
    const int haloElementCount = prms->GridHaloSize*prms->GridY*2;
    const int tpb = prms->tpb_Upd;   // threads per block
    const int blocksPerGrid = (haloElementCount + tpb - 1) / tpb;
    partition_kernel_receive_halos_left<<<blocksPerGrid, tpb, 0, streamCompute>>>(haloElementCount, GridX_partition,
                                                                                  nGridPitch, grid_array,
                                                                                  halo_transfer_buffer[0], halo_transfer_buffer[1]);

    partition_kernel_receive_halos_right<<<blocksPerGrid, tpb, 0, streamCompute>>>(haloElementCount, GridX_partition,
                                                                                   nGridPitch, grid_array,
                                                                                   halo_transfer_buffer[0], halo_transfer_buffer[1]);

    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("receive_halos kernel execution");
}




void GPU_Partition::record_timings(const bool enablePointTransfer)
{
    float _gridResetAndHalo, _acceptHalo, _G2P, _total, _ptsSent, _ptsAccepted;
    float _updateGrid;
    cudaSetDevice(Device);
    cudaError_t err;
    err = cudaStreamSynchronize(streamCompute);

    err = cudaEventElapsedTime(&_gridResetAndHalo, event_10_cycle_start, event_20_grid_halo_sent);
    if(err != cudaSuccess)
    {
        const char *errorString = cudaGetErrorString(err);
        spdlog::error("error string: {}",errorString);
        throw std::runtime_error("record_timings 1");
    }
    err = cudaEventElapsedTime(&_acceptHalo, event_20_grid_halo_sent, event_30_halo_accepted);
    if(err != cudaSuccess) throw std::runtime_error("record_timings 2");
    err = cudaEventElapsedTime(&_updateGrid, event_30_halo_accepted, event_40_grid_updated);
    if(err != cudaSuccess) throw std::runtime_error("record_timings 3");
    err = cudaEventElapsedTime(&_G2P, event_40_grid_updated, event_50_g2p_completed);
    if(err != cudaSuccess) throw std::runtime_error("record_timings 4");

    if(enablePointTransfer)
    {
        err = cudaEventElapsedTime(&_ptsSent, event_50_g2p_completed, event_70_pts_sent);
        if(err != cudaSuccess) throw std::runtime_error("record_timings 6");
        err = cudaEventElapsedTime(&_ptsAccepted, event_70_pts_sent, event_80_pts_accepted);
        if(err != cudaSuccess) throw std::runtime_error("record_timings 7");

        err = cudaEventElapsedTime(&_total, event_10_cycle_start, event_80_pts_accepted);
        if(err != cudaSuccess) throw std::runtime_error("record_timings pts accepted");
    }
    else
    {
        _ptsSent = 0;
        _ptsAccepted = 0;

        err = cudaEventElapsedTime(&_total, event_10_cycle_start, event_50_g2p_completed);
        if(err != cudaSuccess) throw std::runtime_error("record_timings pts accepted");
    }

    timing_10_P2GAndHalo += _gridResetAndHalo;
    timing_20_acceptHalo += _acceptHalo;
    timing_30_updateGrid += _updateGrid;
    timing_40_G2P += _G2P;
    timing_60_ptsSent += _ptsSent;
    timing_70_ptsAccepted += _ptsAccepted;

    timing_stepTotal += _total;

    int left = getLeftBufferCount();
    int right = getRightBufferCount();

    int max_lr = max(left, right);
    max_pts_sent = max(max_pts_sent, max_lr);

    max_pt_deviation = max(max_pt_deviation, getMaxDeviationValue());
}

void GPU_Partition::reset_timings()
{
    max_pts_sent = 0;
    max_pt_deviation = 0;
    timing_10_P2GAndHalo = 0;
    timing_20_acceptHalo = 0;
    timing_30_updateGrid = 0;
    timing_40_G2P = 0;
    timing_60_ptsSent = 0;
    timing_70_ptsAccepted = 0;
    timing_stepTotal = 0;
}

void GPU_Partition::normalize_timings(int cycles)
{
    float coeff = (float)1000/(float)cycles;
    timing_10_P2GAndHalo *= coeff;
    timing_20_acceptHalo *= coeff;
    timing_30_updateGrid *= coeff;
    timing_40_G2P *= coeff;
    timing_60_ptsSent *= coeff;
    timing_70_ptsAccepted *= coeff;
    timing_stepTotal *= coeff;
}

