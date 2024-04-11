#include "gpu_implementation5.h"
#include "parameters_sim.h"
#include "point.h"
#include "model.h"

#include <iostream>
#include <vector>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/LU>

#include <spdlog/spdlog.h>



using namespace Eigen;



void GPU_Implementation5::device_allocate_arrays()
{
    cudaError_t err;
    const unsigned &nPts = model->prms.nPtsTotal;
    const unsigned &nPartitions = model->prms.nPartitions;

    // count available GPUs
    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) throw std::runtime_error("cudaGetDeviceCount error");
    if(deviceCount == 0) throw std::runtime_error("No avaialble CUDA devices");

    partitions.clear();

    int whichDevice = 0;
    unsigned points_counter = 0;
    unsigned grid_x_cell_counter = 0;
    const unsigned points_requested_per_partition = (nPts/nPartitions) * (1 + model->prms.ExtraSpaceForIncomingPoints);
    partitions.resize(nPartitions);
    for(int i=0;i<nPartitions;i++)
    {
        GPU_Partition &partition = partitions[i];
        partition.initialize(whichDevice, i);
        partition.allocate(points_requested_per_partition, model->prms.GridXTotal); // at this time, we allocate the full grid
        whichDevice = (whichDevice+1)%deviceCount;  // spread partitions across the available devices
    }
}

void GPU_Implementation5::transfer_ponts_to_device()
{
    spdlog::info("GPU_Implementation: transfer_to_device() start");
    const double &hinv = model->prms.cellsize_inv;
    const int &GridXTotal = model->prms.GridXTotal;

    unsigned nPointsUploaded = 0;
    const unsigned &nPartitions = model->prms.nPartitions;
    unsigned GridOffset = 0;
    // distribute points except of the last partition
    for(int i=0;i<nPartitions;i++)
    {
        unsigned nPartitionsRemaining = nPartitions - i;
        unsigned tentativePointCount = (hssoa.size - nPointsUploaded)/nPartitionsRemaining;
        int tentativePointIndex = nPointsUploaded + tentativePointCount - 1;
        SOAIterator it2 = hssoa.begin()+tentativePointIndex;
        const ProxyPoint &pt = *it2;
        int cellsIdx = pt.getXIndex(hinv);

        // find the index of the first point with x-index cellsIdx
        unsigned pt_idx;
        if(i==(nPartitions-1)) pt_idx = hssoa.size;
        else
        {
            pt_idx = hssoa.FindFirstPointAtGridXIndex(cellsIdx, hinv);
            partitions[i+1].GridX_offset = cellsIdx;
        }
        // last partition must span to the "end" of the gird along the x-axis
        partitions[i].GridX_partition = i==nPartitions-1 ? GridXTotal-partitions[i].GridX_offset : cellsIdx-partitions[i].GridX_offset;
        partitions[i].nPts_partition = pt_idx-nPointsUploaded;

// TODO: remove this loop if the operation is too slow
#pragma omp parallel for
        for(int j=nPointsUploaded;j<pt_idx;j++)
        {
            SOAIterator it = hssoa.begin()+j;
            it->setPartition((uint8_t)i);
        }

        spdlog::info("transfer partition {}; grid offset {}; grid size {}, npts {}",
                     i, partitions[i].GridX_offset, partitions[i].GridX_partition, partitions[i].nPts_partition);
        partitions[i].transfer_points_from_soa_to_device(hssoa, nPointsUploaded);
        nPointsUploaded = pt_idx;
    }

    for(int i=0;i<partitions.size();i++) partitions[i].clear_utility_vectors();
    spdlog::info("transfer_ponts_to_device() done; uploaded {}",nPointsUploaded);
}



void GPU_Implementation5::transfer_from_device()
{
    unsigned offset_pts = 0;
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        p.transfer_from_device(hssoa, offset_pts);
        offset_pts += p.nPts_partition;
    }
    hssoa.size = offset_pts;

    // wait until everything is copied to host
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        cudaSetDevice(p.Device);
        cudaStreamSynchronize(p.streamCompute);
        if(p.error_code)
        {
            spdlog::critical("P {}; error code {}", p.PartitionID, p.error_code);
            throw std::runtime_error("error code");
        }
    }

    if(transfer_completion_callback) transfer_completion_callback();
}


void GPU_Implementation5::synchronize()
{
    for(GPU_Partition &p : partitions)
    {
        cudaSetDevice(p.Device);
        cudaDeviceSynchronize();
    }
}

void GPU_Implementation5::update_constants()
{
    for(GPU_Partition &p : partitions) p.update_constants();
}


void GPU_Implementation5::reset_grid()
{
    for(GPU_Partition &p : partitions) p.reset_grid();
}

void GPU_Implementation5::reset_indenter_force_accumulator()
{
    for(GPU_Partition &p : partitions) p.reset_indenter_force_accumulator();
}



void GPU_Implementation5::p2g()
{
    cudaError_t err;
    const size_t haloSize = model->prms.GridHaloSize*sizeof(double)*model->prms.GridY;
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        p.p2g();
/*
        if(i!=(partitions.size()-1))
        {
            GPU_Partition &pnxt = partitions[i+1];
            for(int j=0;j<icy::SimParams::nGridArrays;j++)
            {
                double *halo_src1 = p.getHaloAddress(1, j);
                double *halo_dst1 = pnxt.getHaloReceiveAddress(0, j);
                err = cudaMemcpyPeerAsync(halo_dst1, pnxt.Device, halo_src1, p.Device, haloSize, p.streamCompute);
                if(err != cudaSuccess) throw std::runtime_error("p2g cudaMemcpyPeerAsync");
            }
        }
        if(i!=0)
        {
            GPU_Partition &pprev = partitions[i-1];
            for(int j=0;j<icy::SimParams::nGridArrays;j++)
            {
                double *halo_src1 = p.getHaloAddress(0, j);
                double *halo_dst1 = pprev.getHaloReceiveAddress(1, j);
                err = cudaMemcpyPeerAsync(halo_dst1, pprev.Device, halo_src1, p.Device, haloSize, p.streamCompute);
                if(err != cudaSuccess) throw std::runtime_error("p2g cudaMemcpyPeerAsync");
            }
        }
*/
        err = cudaEventRecord(p.event_grid_halo_sent, p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("p2g");
    }
}


void GPU_Implementation5::receive_halos()
{
    cudaError_t err;
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        cudaSetDevice(p.Device);
        if(i!=0)
        {
            GPU_Partition &pprev = partitions[i-1];
            err = cudaStreamWaitEvent(p.streamCompute, pprev.event_grid_halo_sent);
            if(err != cudaSuccess) throw std::runtime_error("receive_halos waiting on event");
        }
        if(i!=partitions.size()-1)
        {
            GPU_Partition &pnxt = partitions[i+1];
            err = cudaStreamWaitEvent(p.streamCompute, pnxt.event_grid_halo_sent);
            if(err != cudaSuccess) throw std::runtime_error("receive_halos waiting on event");
        }
        p.receive_halos();
    }
}


void GPU_Implementation5::update_nodes()
{
    for(GPU_Partition &p : partitions) p.update_nodes();
}


void GPU_Implementation5::g2p(bool recordPQ)
{
    for(GPU_Partition &p : partitions) p.g2p(recordPQ);
}

void GPU_Implementation5::receive_points()
{
    cudaError_t err;
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        err = cudaSetDevice(p.Device);
        if(err != cudaSuccess) throw std::runtime_error("RP cudaSetDevice");
//        err = cudaEventSynchronize(p.event_utility_data_transferred);
//        if(err != cudaSuccess) throw std::runtime_error("RP cudaEventSynchronize");
        err = cudaStreamSynchronize(p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("RP cudaStreamSynchronize");

        if(i!=(partitions.size()-1))
        {
            // send buffer to the right
            GPU_Partition &pnxt = partitions[i+1];
            double *src_point_buffer = p.point_transfer_buffer[1];
            double *dst_point_buffer = pnxt.point_transfer_buffer[2];
            size_t count = p.getRightBufferCount()*sizeof(double)*icy::SimParams::nPtsArrays;
            err = cudaMemcpyPeerAsync(dst_point_buffer, pnxt.Device, src_point_buffer, p.Device, count, p.streamCompute);
            if(err != cudaSuccess) throw std::runtime_error("RP cudaMemcpyPeerAsync");
        }
        if(i!=0)
        {
            // send buffer to the right
            GPU_Partition &pprev = partitions[i-1];
            double *src_point_buffer = p.point_transfer_buffer[0];
            double *dst_point_buffer = pprev.point_transfer_buffer[3];
            size_t count = p.getLeftBufferCount()*sizeof(double)*icy::SimParams::nPtsArrays;
            err = cudaMemcpyPeerAsync(dst_point_buffer, pprev.Device, src_point_buffer, p.Device, count, p.streamCompute);
            if(err != cudaSuccess) throw std::runtime_error("RP cudaMemcpyPeerAsync");
        }
        err = cudaEventRecord(p.event_pts_sent, p.streamCompute);
        if(err != cudaSuccess)
        {
            const char* errorString = cudaGetErrorString(err);
            spdlog::critical("RP error {}: {}", err, errorString);
            throw std::runtime_error("RP");
        }
        //spdlog::info("RP P {}; left {}; right {}", p.PartitionID, p.getLeftBufferCount(), p.getRightBufferCount());
    }

    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        cudaSetDevice(p.Device);
        unsigned left=0, right=0;
        if(i!=0)
        {
            GPU_Partition &pprev = partitions[i-1];
            err = cudaStreamWaitEvent(p.streamCompute, partitions[i-1].event_pts_sent);
            if(err != cudaSuccess) throw std::runtime_error("RP wait event");
            left = pprev.getRightBufferCount();
        }
        if(i!=partitions.size()-1)
        {
            GPU_Partition &pnxt = partitions[i+1];
            err = cudaStreamWaitEvent(p.streamCompute, partitions[i+1].event_pts_sent);
            if(err != cudaSuccess) throw std::runtime_error("RP wait event");
            right = pnxt.getLeftBufferCount();
        }
        p.receive_points(left, right);
    }
}








/*

// ==============================  kernels  ====================================

__device__ Matrix2d polar_decomp_R(const Matrix2d &val)
{
    // polar decomposition
    // http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
    double th = atan2(val(1,0) - val(0,1), val(0,0) + val(1,1));
    Matrix2d result;
    result << cosf(th), -sinf(th), sinf(th), cosf(th);
    return result;
}

// ========================================= initialization and kernel execution

void CUDART_CB GPU_Implementation5::callback_from_stream(cudaStream_t stream, cudaError_t status, void *userData)
{
    // simulation data was copied to host memory -> proceed with processing of this data
    GPU_Implementation5 *gpu = reinterpret_cast<GPU_Implementation5*>(userData);
    // any additional processing here
    if(gpu->transfer_completion_callback) gpu->transfer_completion_callback();
}
*/
