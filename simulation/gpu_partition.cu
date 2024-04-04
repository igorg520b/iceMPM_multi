#include "gpu_partition.h"

using namespace Eigen;

constexpr double d = 2; // dimensions
constexpr double coeff1 = 1.4142135623730950; // sqrt((6-d)/2.);

__device__ uint8_t gpu_error_indicator;
__constant__ icy::SimParams gprms;

icy::SimParams *GPU_Partition::prms;

double* GPU_Partition::getHaloAddress(int whichHalo, int whichGridArray)
{
    if(whichHalo == 0)
    {
        // left halo
        return grid_array + (prms->GridY * prms->GridHaloSize*2) + whichGridArray*nGridPitch;
    }
    else if(whichHalo == 1)
    {
        // right halo
        return grid_array + prms->GridY * (GridX_partition + 3*prms->GridHaloSize) + whichGridArray*nGridPitch;
    }
    else throw std::runtime_error("getHaloAddress");
}

double* GPU_Partition::getHaloReceiveAddress(int whichHalo, int whichGridArray)
{
    if(whichHalo == 0)
    {
        // left halo
        return grid_array + (prms->GridY * prms->GridHaloSize*0) + whichGridArray*nGridPitch;
    }
    else if(whichHalo == 1)
    {
        // right halo
        return grid_array + (prms->GridY * prms->GridHaloSize*1) + whichGridArray*nGridPitch;
    }
    else throw std::runtime_error("getHaloReceiveAddress");
}



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


void GPU_Partition::p2g()
{
    cudaSetDevice(Device);
    const unsigned &n = nPts_partition;
    const unsigned &tpb = prms->tpb_P2G;
    const unsigned blocksPerGrid = (n + tpb - 1) / tpb;
    partition_kernel_p2g<<<blocksPerGrid, tpb, 0, streamCompute>>>(GridX_partition, GridX_offset, nGridPitch,
                         nPts_partition, nPtsPitch, pts_array, grid_array);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("p2g kernel");
}


__global__ void partition_kernel_p2g(const unsigned gridX, const unsigned gridX_offset, const unsigned pitch_grid,
                              const unsigned count_pts, const unsigned pitch_pts,
                              const double *buffer_pts, double *buffer_grid)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= count_pts) return;

    const long long* ptr = reinterpret_cast<const long long*>(&buffer_pts[pitch_pts*icy::SimParams::idx_utility_data]);
    long long utility_data = ptr[pt_idx];
    if(utility_data & 0x20000) return; // point is disabled

    const double &dt = gprms.InitialTimeStep;
    const double &vol = gprms.ParticleVolume;
    const double &h = gprms.cellsize;
    const double &h_inv = gprms.cellsize_inv;
    const double &Dinv = gprms.Dp_inv;
    const double &particle_mass = gprms.ParticleMass;

    const unsigned &gridY = gprms.GridY;
    const int &halo = gprms.GridHaloSize;

    // pull point data from SOA
    Vector2d pos, velocity;
    Matrix2d Bp, Fe;

    for(int i=0; i<icy::SimParams::dim; i++)
    {
        pos[i] = buffer_pts[pt_idx + pitch_pts*(icy::SimParams::posx+i)];
        velocity[i] = buffer_pts[pt_idx + pitch_pts*(icy::SimParams::velx+i)];
        for(int j=0; j<icy::SimParams::dim; j++)
        {
            Fe(i,j) = buffer_pts[pt_idx + pitch_pts*(icy::SimParams::Fe00 + i*icy::SimParams::dim + j)];
            Bp(i,j) = buffer_pts[pt_idx + pitch_pts*(icy::SimParams::Bp00 + i*icy::SimParams::dim + j)];
        }
    }

    Matrix2d PFt = KirchhoffStress_Wolper(Fe);
    Matrix2d subterm2 = particle_mass*Bp - (gprms.dt_vol_Dpinv)*PFt;

    Eigen::Vector2i base_coord_i = (pos*h_inv - Vector2d::Constant(0.5)).cast<int>(); // coords of base grid node for point
    Vector2d base_coord = base_coord_i.cast<double>();
    Vector2d fx = pos*h_inv - base_coord;

    // optimized method of computing the quadratic (!) weight function (no conditional operators)
    Array2d arr_v0 = 1.5-fx.array();
    Array2d arr_v1 = fx.array() - 1.0;
    Array2d arr_v2 = fx.array() - 0.5;
    Array2d ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            double Wip = ww[i][0]*ww[j][1];
            Vector2d dpos((i-fx[0])*h, (j-fx[1])*h);
            Vector2d incV = Wip*(velocity*particle_mass + subterm2*dpos);
            double incM = Wip*particle_mass;

            // the x-index of the cell takes into accout the partition's offset of the gird fragment
            int i2 = i+base_coord_i[0]-gridX_offset;
            int j2 = j+base_coord_i[1];
            if(i2<(-halo) || j2<0 || i2>=(gridX+halo) || j2>=gridY) gpu_error_indicator = 1;
            int idx_gridnode = j2 + (i2+halo*3)*gridY;  // two halo lines are reserved for the incoming halo data

            // Udpate mass, velocity and force
            atomicAdd(&buffer_grid[0*pitch_grid + idx_gridnode], incM);
            atomicAdd(&buffer_grid[1*pitch_grid + idx_gridnode], incV[0]);
            atomicAdd(&buffer_grid[2*pitch_grid + idx_gridnode], incV[1]);
        }
}


__device__ Matrix2d KirchhoffStress_Wolper(const Matrix2d &F)
{
    const double &kappa = gprms.kappa;
    const double &mu = gprms.mu;

    // Kirchhoff stress as per Wolper (2019)
    double Je = F.determinant();
    Matrix2d b = F*F.transpose();
    Matrix2d PFt = mu*(1/Je)*dev(b) + kappa*(Je*Je-1.)*Matrix2d::Identity();
    return PFt;
}

// deviatoric part of a diagonal matrix
__device__ Vector2d dev_d(Vector2d Adiag)
{
    return Adiag - Adiag.sum()/2*Vector2d::Constant(1.);
}

__device__ Eigen::Matrix2d dev(Eigen::Matrix2d A)
{
    return A - A.trace()/2*Eigen::Matrix2d::Identity();
}

