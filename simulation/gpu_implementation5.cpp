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


void GPU_Implementation5::reset_grid()
{
    cudaError_t err;
    for(GPU_Partition &p : partitions)
    {
        err = cudaSetDevice(p.Device);
        if(err != cudaSuccess) throw std::runtime_error("reset_grid set device");
        err = cudaEventRecord(p.event_10_cycle_start, p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("reset_grid event");
        p.reset_grid();
    }
}

void GPU_Implementation5::p2g()
{
    cudaError_t err;
    const int &halo = model->prms.GridHaloSize;
    const int &gridY = model->prms.GridY;
    const int &gridXT = model->prms.GridXTotal;
    const int &offset = model->prms.gbOffset;


    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        p.p2g();    // invoke the P2G kernel

        if(i!=0)
        {
            // send halo to the left
            GPU_Partition &pprev = partitions[i-1];
            for(int j=0;j<icy::SimParams::nGridArrays;j++)
            {
                double *src = p.grid_array + j*p.nGridPitch;
                double *dst = pprev.halo_transfer_buffer[1] + j*pprev.nGridPitch;
                const size_t halo_count = sizeof(double)*gridY*halo*2;
                err = cudaMemcpyPeerAsync(dst, pprev.Device, src, p.Device, halo_count, p.streamCompute);
            }
            if(err != cudaSuccess) throw std::runtime_error("p2g cudaMemcpyPeerAsync");
        }

        if(i!=(partitions.size()-1))
        {
            // send halo to the right

            GPU_Partition &pnxt = partitions[i+1];
            for(int j=0;j<icy::SimParams::nGridArrays;j++)
            {
                double *src = p.grid_array + j*p.nGridPitch + gridY*(p.GridX_partition);
                double *dst = pnxt.halo_transfer_buffer[0] + j*pnxt.nGridPitch;
                const size_t halo_count = sizeof(double)*gridY*halo*2;
                err = cudaMemcpyPeerAsync(dst, pnxt.Device, src, p.Device, halo_count, p.streamCompute);
            }
            if(err != cudaSuccess) throw std::runtime_error("p2g cudaMemcpyPeerAsync");
        }

        err = cudaEventRecord(p.event_20_grid_halo_sent, p.streamCompute);
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
            err = cudaStreamWaitEvent(p.streamCompute, pprev.event_20_grid_halo_sent);
            if(err != cudaSuccess) throw std::runtime_error("receive_halos waiting on event");
        }
        if(i!=partitions.size()-1)
        {
            GPU_Partition &pnxt = partitions[i+1];
            err = cudaStreamWaitEvent(p.streamCompute, pnxt.event_20_grid_halo_sent);
            if(err != cudaSuccess) throw std::runtime_error("receive_halos waiting on event");
        }
        p.receive_halos();

        cudaError_t err = cudaEventRecord(p.event_30_halo_accepted, p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("receive_halos event");
    }
}


void GPU_Implementation5::update_nodes()
{
    for(GPU_Partition &p : partitions)
    {
        p.update_nodes();
        cudaError_t err = cudaEventRecord(p.event_40_grid_updated, p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("update_nodes cudaEventRecord");
    }
}

void GPU_Implementation5::g2p(const bool recordPQ, const bool enablePointTransfer)
{
    for(GPU_Partition &p : partitions)
    {
        p.g2p(recordPQ, enablePointTransfer);
        cudaError_t err = cudaEventRecord(p.event_50_g2p_completed, p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("g2p cudaEventRecord");
    }
}

void GPU_Implementation5::receive_points()
{
    cudaError_t err;
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        err = cudaSetDevice(p.Device);
        if(err != cudaSuccess) throw std::runtime_error("RP cudaSetDevice");

        err = cudaEventSynchronize(p.event_50_g2p_completed); // unfortunately, we must wait until host_utility_data populates
        if(err != cudaSuccess) throw std::runtime_error("RP cudaEventSynchronize");
        p.nPts_disabled += (p.getRightBufferCount()+p.getLeftBufferCount());


        if(i!=(partitions.size()-1))
        {
            // send buffer to the right
            GPU_Partition &pnxt = partitions[i+1];
            double *src_point_buffer = p.point_transfer_buffer[1];
            double *dst_point_buffer = pnxt.point_transfer_buffer[2];
            int right_buffer_count = p.getRightBufferCount();
            size_t count = right_buffer_count*sizeof(double)*icy::SimParams::nPtsArrays;
            if(count != 0)
            {
                err = cudaMemcpyPeerAsync(dst_point_buffer, pnxt.Device, src_point_buffer, p.Device, count, p.streamCompute);
                if(err != cudaSuccess) throw std::runtime_error("RP copy buffer to the right");
            }
        }
        if(i!=0)
        {
            // send buffer to the right
            GPU_Partition &pprev = partitions[i-1];
            double *src_point_buffer = p.point_transfer_buffer[0];
            double *dst_point_buffer = pprev.point_transfer_buffer[3];
            int left_buffer_count = p.getLeftBufferCount();
            size_t count = left_buffer_count*sizeof(double)*icy::SimParams::nPtsArrays;
            if(count != 0)
            {
                err = cudaMemcpyPeerAsync(dst_point_buffer, pprev.Device, src_point_buffer, p.Device, count, p.streamCompute);
                if(err != cudaSuccess) throw std::runtime_error("RP copy buffer to the left");
            }
        }
        err = cudaEventRecord(p.event_70_pts_sent, p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("RP cudaEventRecord");
    }

    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        cudaSetDevice(p.Device);
        int left=0, right=0;
        if(i!=0)
        {
            GPU_Partition &pprev = partitions[i-1];
            err = cudaStreamWaitEvent(p.streamCompute, partitions[i-1].event_70_pts_sent);
            if(err != cudaSuccess) throw std::runtime_error("RP wait event");
            left = pprev.getRightBufferCount();
        }
        if(i!=partitions.size()-1)
        {
            GPU_Partition &pnxt = partitions[i+1];
            err = cudaStreamWaitEvent(p.streamCompute, partitions[i+1].event_70_pts_sent);
            if(err != cudaSuccess) throw std::runtime_error("RP wait event");
            right = pnxt.getLeftBufferCount();
        }
        p.receive_points(left, right);
        cudaError_t err = cudaEventRecord(p.event_80_pts_accepted, p.streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("receive_points event record");
    }
}




void GPU_Implementation5::record_timings(const bool enablePointTransfer)
{
    for(GPU_Partition &p : partitions) p.record_timings(enablePointTransfer);
}



// ==========================================================================




void GPU_Implementation5::initialize_and_enable_peer_access()
{
    const int &nPartitions = model->prms.nPartitions;

    // count available GPUs
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) throw std::runtime_error("cudaGetDeviceCount error");
    if(deviceCount == 0) throw std::runtime_error("No avaialble CUDA devices");

    partitions.clear();
    partitions.resize(nPartitions);

    for(int i=0;i<nPartitions;i++)
    {
        GPU_Partition &p = partitions[i];
        p.initialize(i%deviceCount, i);
    }

    for(int i=0;i<nPartitions;i++)
    {
        GPU_Partition &p = partitions[i];
        err = cudaSetDevice(p.Device);
        if(err != cudaSuccess) throw std::runtime_error("initialize_and_enable_peer_access cudaSetDevice");

        if(i!=0)
        {
            GPU_Partition &pprev = partitions[i-1];
            // enable access to device on the left
            if(p.Device != pprev.Device)
            {
                err = cudaDeviceEnablePeerAccess(pprev.Device, 0);
                if(err != cudaSuccess)
                {
                    spdlog::error("P{}: err {}; pprev.Device {} ; p.Device {}", p.PartitionID, err, pprev.Device, p.Device );
                    throw std::runtime_error("GPU_Implementation5::device_allocate_arrays() cudaDeviceEnablePeerAccess L");
                }
            }
        }
        if(i!=nPartitions-1)
        {
            GPU_Partition &pnxt = partitions[i+1];
            // enable access to device on the right
            if(p.Device != pnxt.Device)
            {
                err = cudaDeviceEnablePeerAccess(pnxt.Device, 0);
                if(err != cudaSuccess)
                {
                    spdlog::error("P{}: err {}; pnxt.Device {} ; p.Device {}", p.PartitionID, err, pnxt.Device, p.Device);
                    throw std::runtime_error("GPU_Implementation5::device_allocate_arrays() cudaDeviceEnablePeerAccess R");
                }
            }
        }
    }
}


void GPU_Implementation5::split_hssoa_into_partitions()
{
    spdlog::info("split_hssoa_into_partitions() start");
    const double &hinv = model->prms.cellsize_inv;
    const int &GridXTotal = model->prms.GridXTotal;
    const int &nPartitions = model->prms.nPartitions;

    unsigned nPointsProcessed = 0;
    unsigned GridOffset = 0;

    for(int i=0;i<nPartitions;i++)
    {
        GPU_Partition &p = partitions[i];
        p.nPts_disabled = 0;
        const int nPartitionsRemaining = nPartitions - i;
        p.nPts_partition = (hssoa.size - nPointsProcessed)/nPartitionsRemaining; // points in this partition

        // find the index of the first point with x-index cellsIdx
        if(i < nPartitions-1)
        {
            SOAIterator it2 = hssoa.begin() + (nPointsProcessed + p.nPts_partition);
            const int cellsIdx = it2->getXIndex(hinv);
            p.GridX_partition = cellsIdx - p.GridX_offset;
            partitions[i+1].GridX_offset = cellsIdx;
        }
        else if(i == nPartitions-1)
        {
            // the last partition spans the rest of the grid along the x-axis
            p.GridX_partition = GridXTotal - p.GridX_offset;
        }

#pragma omp parallel for
        for(int j=nPointsProcessed;j<(nPointsProcessed+p.nPts_partition);j++)
        {
            SOAIterator it = hssoa.begin()+j;
            it->setPartition((uint8_t)i);
        }

        spdlog::info("split: P {}; grid_offset {}; grid_size {}, npts {}",
                     i, partitions[i].GridX_offset, partitions[i].GridX_partition, partitions[i].nPts_partition);
        nPointsProcessed += p.nPts_partition;
    }
}


void GPU_Implementation5::allocate_arrays()
{
    cudaError_t err;
    const unsigned &nPts = model->prms.nPtsTotal;

    auto it = std::max_element(partitions.begin(), partitions.end(),
                               [](const GPU_Partition &p1, const GPU_Partition &p2)
                               {return p1.GridX_partition < p2.GridX_partition;});
    int max_GridX_size = it->GridX_partition;
    int GridX_size = std::min(max_GridX_size*2, model->prms.GridXTotal);

    const unsigned points_requested_per_partition = (nPts/partitions.size()) * (1 + model->prms.ExtraSpaceForIncomingPoints);
    for(GPU_Partition &p : partitions) p.allocate(points_requested_per_partition, GridX_size);
}



void GPU_Implementation5::transfer_ponts_to_device()
{
    spdlog::info("GPU_Implementation: transfer_to_device()");
    int points_uploaded = 0;
    for(GPU_Partition &p : partitions)
    {
        p.transfer_points_from_soa_to_device(hssoa, points_uploaded);
        points_uploaded += p.nPts_partition;
    }
    spdlog::info("transfer_ponts_to_device() done; transferred points {}", points_uploaded);
}



void GPU_Implementation5::transfer_from_device()
{
    unsigned offset_pts = 0;
    for(int i=0;i<partitions.size();i++)
    {
        GPU_Partition &p = partitions[i];
        int capacity_required = offset_pts + p.nPts_partition;
        if(capacity_required > hssoa.capacity)
        {
            spdlog::error("transfer_from_device(): capacity {} exceeded ({}) when transferring P {}",
                             hssoa.capacity, capacity_required, p.PartitionID);
            throw std::runtime_error("transfer_from_device capacity exceeded");
        }

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

    int count = 0;
    for(int i=0;i<hssoa.size;i++)
    {
        SOAIterator s = hssoa.begin()+i;
        if(s->getDisabledStatus()) continue;
        count++;
    }

    if(count != model->prms.nPtsTotal)
    {
        spdlog::error("tranfer: hssoa.size {}; nPts {}, count_active {}", hssoa.size, model->prms.nPtsTotal, count);

        unsigned offset_pts = 0;
        for(int i=0;i<partitions.size();i++)
        {
            GPU_Partition &p = partitions[i];
            int capacity_required = offset_pts + p.nPts_partition;
            int count_disabled_soa = 0;
            for(int i=offset_pts; i<offset_pts+p.nPts_partition; i++)
            {
                SOAIterator s = hssoa.begin()+i;
                if(s->getDisabledStatus())
                {
                    std::cout << i << ' ';
                    count_disabled_soa++;
                }
            }
            std::cout << '\n';
            offset_pts += p.nPts_partition;
            spdlog::error("P{}: size {}; disabled {}; disabled_soa {}",
                          p.PartitionID, p.nPts_partition, p.nPts_disabled, count_disabled_soa);
        }


        throw std::runtime_error("transfer_from_device(): active point count mismatch");
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

void GPU_Implementation5::reset_indenter_force_accumulator()
{
    for(GPU_Partition &p : partitions)
    {
        p.reset_indenter_force_accumulator();
        p.reset_timings();
    }
}






// ==============================================================================









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
