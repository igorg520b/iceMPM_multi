#include "gpu_partition.h"
#include "helper_math.cuh"
#include <stdio.h>

using namespace Eigen;

constexpr double d = 2; // dimensions
constexpr double coeff1 = 1.4142135623730950; // sqrt((6-d)/2.);
constexpr long long status_crushed = 0x10000;
constexpr long long status_disabled = 0x20000;

__device__ uint8_t gpu_error_indicator;
__constant__ icy::SimParams gprms;

icy::SimParams *GPU_Partition::prms;


void GPU_Partition::receive_nodes(unsigned nFromLeft, unsigned nFromRight)
{
    unsigned total = nFromLeft+nFromRight;
    if(!total) return;

/*
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("g2p cudaSetDevice");

    err = cudaMemsetAsync(device_side_utility_data, 0, 2*sizeof(unsigned), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("g2p cudaMemset");

    const unsigned &n = nPts_partition;
    const unsigned &tpb = prms->tpb_G2P;
    const unsigned nBlocks = (n + tpb - 1) / tpb;
*/
}


__global__ void partition_kernel_receive_points(const unsigned count_pts, const unsigned pitch_pts,
                                                double *buffer_pts, const double *buffer_grid,
                                                double *_point_transfer_buffer[4], unsigned *vector_data_disabled_points,
                                                size_t VectorCapacity_transfer, size_t VectorCapacity_disabled)
{

}



void GPU_Partition::g2p(const bool recordPQ)
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("g2p cudaSetDevice");

    err = cudaMemsetAsync(device_side_utility_data, 0, 2*sizeof(unsigned), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("g2p cudaMemset");

    const unsigned &n = nPts_partition;
    const unsigned &tpb = prms->tpb_G2P;
    const unsigned nBlocks = (n + tpb - 1) / tpb;


    partition_kernel_g2p<<<nBlocks, tpb, 0, streamCompute>>>(recordPQ,
            GridX_partition, GridX_offset, nGridPitch,
            nPts_partition, nPtsPitch,
            pts_array, grid_array,
            point_transfer_buffer, vector_data_disabled_points, device_side_utility_data,
            prms->VectorCapacity_transfer, prms->VectorCapacity_disabled);

    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("g2p kernel");

    err = cudaMemcpyAsync(host_side_utility_data, device_side_utility_data, sizeof(unsigned)*utility_data_size, cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess)
    {
        const char* errorString = cudaGetErrorString(err);
        spdlog::critical("error {}; host {}; device {}",errorString, (void*)host_side_utility_data, (void*)device_side_utility_data);
        throw std::runtime_error("GPU_Partition::g2p cudaMemcpy");
    }
    err = cudaEventRecord(event_utility_data_transferred, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition::g2p cudaEventRecord");
}

__global__ void partition_kernel_g2p(const bool recordPQ,
                                    const int gridX, const int gridX_offset, const unsigned pitch_grid,
                                    const unsigned count_pts, const unsigned pitch_pts,
                                    double *buffer_pts, const double *buffer_grid,
                                    double *_point_transfer_buffer[4], unsigned *vector_data_disabled_points,
                                    unsigned *utility_data,
                                    const unsigned VectorCapacity_transfer, const unsigned VectorCapacity_disabled)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= count_pts) return;

    // skip if a point is disabled
    icy::Point p;
    long long* ptr = reinterpret_cast<long long*>(&buffer_pts[pitch_pts*icy::SimParams::idx_utility_data]);
    p.utility_data = ptr[pt_idx];
    if(p.utility_data & status_disabled) return; // point is disabled

    const int &halo = gprms.GridHaloSize;
    const double &h_inv = gprms.cellsize_inv;
//    const double &h = gprms.cellsize;
    const double &dt = gprms.InitialTimeStep;
    const int &gridY = gprms.GridY;
    const double &mu = gprms.mu;
    const double &kappa = gprms.kappa;

    p.velocity.setZero();
    p.Bp.setZero();

    // pull point data from SOA
    for(int i=0; i<icy::SimParams::dim; i++)
    {
        p.pos[i] = buffer_pts[pt_idx + pitch_pts*(icy::SimParams::posx+i)];
        for(int j=0; j<icy::SimParams::dim; j++)
        {
            p.Fe(i,j) = buffer_pts[pt_idx + pitch_pts*(icy::SimParams::Fe00 + i*icy::SimParams::dim + j)];
        }
    }
    p.Jp_inv = buffer_pts[pt_idx + pitch_pts*icy::SimParams::idx_Jp_inv];
    p.grain = (short)p.utility_data;
    p.crushed = (p.utility_data >> 16) & 0x1;

    // coords of base grid node for point
    Eigen::Vector2i base_coord_i = (p.pos*h_inv - Vector2d::Constant(0.5)).cast<int>();
    Vector2d base_coord = base_coord_i.cast<double>();
    Vector2d fx = p.pos*h_inv - base_coord;

    // optimized method of computing the quadratic weight function without conditional operators
    Array2d arr_v0 = 1.5 - fx.array();
    Array2d arr_v1 = fx.array() - 1.0;
    Array2d arr_v2 = fx.array() - 0.5;
    Array2d ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            Vector2d dpos = Vector2d(i, j) - fx;
            double weight = ww[i][0]*ww[j][1];

            int i2 = i+base_coord_i[0]-gridX_offset;
            int j2 = j+base_coord_i[1];
            int idx_gridnode = j2 + (i2+halo*3)*gridY;  // two halo lines are reserved for the incoming halo data

            Vector2d node_velocity;
            node_velocity[0] = buffer_grid[1*pitch_grid + idx_gridnode];
            node_velocity[1] = buffer_grid[2*pitch_grid + idx_gridnode];
            p.velocity += weight * node_velocity;
            p.Bp += (4.*h_inv)*weight *(node_velocity*dpos.transpose());
        }

    // Advection and update of the deformation gradient
    p.pos += p.velocity * dt;
    p.Fe = (Matrix2d::Identity() + dt*p.Bp) * p.Fe;     // p.Bp is the gradient of the velocity vector (it seems)

    ComputePQ(p, kappa, mu);    // pre-computes USV, p, q, etc.

//    if(p.crushed == 0) CheckIfPointIsInsideFailureSurface(p);
    if(__builtin_expect(p.crushed, 0) == 0) CheckIfPointIsInsideFailureSurface(p);
    if(p.crushed == 1) Wolper_Drucker_Prager(p);

    // if a point is about to move to another GPU partition/device, disable it and store in a special array
    base_coord_i = (p.pos*h_inv - Vector2d::Constant(0.5)).cast<int>(); // update the base node index
    if(base_coord_i.x() < gridX_offset-halo)
    {
        // point transfers to the left
        PreparePointForTransfer(pt_idx, 0, _point_transfer_buffer, vector_data_disabled_points, utility_data, p,
                                VectorCapacity_transfer, VectorCapacity_disabled);
    }
    else if(base_coord_i.x() > (gridX_offset+gridX-3+halo))
    {
        // point transfers to the right
        PreparePointForTransfer(pt_idx, 1, _point_transfer_buffer, vector_data_disabled_points, utility_data, p,
                                VectorCapacity_transfer, VectorCapacity_disabled);
    }
    // distribute the values of p back into GPU memory: pos, velocity, BP, Fe, Jp_inv, PQ
    for(int i=0; i<icy::SimParams::dim; i++)
    {
        buffer_pts[pt_idx + pitch_pts*(icy::SimParams::posx+i)] = p.pos[i];
        buffer_pts[pt_idx + pitch_pts*(icy::SimParams::velx+i)] = p.velocity[i];
        for(int j=0; j<icy::SimParams::dim; j++)
        {
            buffer_pts[pt_idx + pitch_pts*(icy::SimParams::Fe00 + i*icy::SimParams::dim + j)] = p.Fe(i,j);
            buffer_pts[pt_idx + pitch_pts*(icy::SimParams::Bp00 + i*icy::SimParams::dim + j)] = p.Bp(i,j);
        }
    }

    buffer_pts[pt_idx + pitch_pts*icy::SimParams::idx_Jp_inv] = p.Jp_inv;
    ptr[pt_idx] = p.utility_data; // includes crushed/disable status and grain number

    // at the end of each cycle, PQ are recorded for visualization
    if(recordPQ)
    {
        buffer_pts[pt_idx + pitch_pts*icy::SimParams::idx_P] = p.p_tr;
        buffer_pts[pt_idx + pitch_pts*icy::SimParams::idx_Q] = p.q_tr;
    }
}

__device__ void PreparePointForTransfer(const unsigned pt_idx, const int whichSide, double *_point_transfer_buffer[4],
                                        unsigned *vector_data_disabled_points, unsigned *utility_data,
                                        icy::Point &p,
                                        const unsigned VectorCapacity_transfer, const unsigned VectorCapacity_disabled)
{
    unsigned fly_idx=0, disabled_buffer_idx=0;

    fly_idx = atomicAdd(&utility_data[whichSide], (unsigned)1);  // reserve space in the transfer buffer
    disabled_buffer_idx = atomicAdd(&utility_data[2], (unsigned)1); // keep track of disabled points

    // check buffer boundary
    if(fly_idx >= VectorCapacity_transfer) { gpu_error_indicator = 2; return; }
    if(disabled_buffer_idx >= VectorCapacity_disabled) { gpu_error_indicator = 3; return; }

    // add the disabled index to the special index table
    vector_data_disabled_points[disabled_buffer_idx] = pt_idx;    // current point storage space is added to the list

    // location where point data will be written
    double *buffer = _point_transfer_buffer[whichSide]+fly_idx*icy::SimParams::nPtsArrays;

    // copy point data into the transfer buffer
    long long *ptr = reinterpret_cast<long long*>(buffer);
    *ptr = p.utility_data;
    for(int i=0; i<icy::SimParams::dim; i++)
    {
        buffer[icy::SimParams::posx+i] = p.pos[i];
        buffer[icy::SimParams::velx+i] = p.velocity[i];
        for(int j=0; j<icy::SimParams::dim; j++)
        {
            buffer[icy::SimParams::Fe00 + i*icy::SimParams::dim + j] = p.Fe(i,j);
            buffer[icy::SimParams::Bp00 + i*icy::SimParams::dim + j] = p.Bp(i,j);
        }
    }
    buffer[icy::SimParams::idx_Jp_inv] = p.Jp_inv;
    buffer[icy::SimParams::idx_P] = p.p_tr;
    buffer[icy::SimParams::idx_Q] = p.q_tr;

    p.utility_data |= status_disabled;    // disable the point on current partition
}


__device__ void svd(const double a[4], double u[4], double sigma[2], double v[4])
{
    GivensRotation<double> gv(0, 1);
    GivensRotation<double> gu(0, 1);
    singular_value_decomposition(a, gu, sigma, gv);
    gu.template fill<2, double>(u);
    gv.template fill<2, double>(v);
}

__device__ void svd2x2(const Matrix2d &mA, Matrix2d &mU, Vector2d &mS, Matrix2d &mV)
{
    double U[4], V[4], S[2];
    double a[4] = {mA(0,0), mA(0,1), mA(1,0), mA(1,1)};
    svd(a, U, S, V);
    mU << U[0],U[1],U[2],U[3];
    mS << S[0],S[1];
    mV << V[0],V[1],V[2],V[3];
}

__device__ void ComputePQ(icy::Point &p, const double &kappa, const double &mu)
{
    svd2x2(p.Fe, p.U, p.vSigma, p.V);
    p.Je_tr = p.vSigma.prod();         // product of elements of vSigma (representation of diagonal matrix)
    p.p_tr = -(kappa/2.) * (p.Je_tr*p.Je_tr - 1.);
    p.vSigmaSquared = p.vSigma.array().square().matrix();
    p.v_s_hat_tr = mu/p.Je_tr * dev_d(p.vSigmaSquared); //mu * pow(Je_tr,-2./d)* dev(SigmaSquared);
    p.q_tr = coeff1*p.v_s_hat_tr.norm();
}


__device__ void Wolper_Drucker_Prager(icy::Point &p)
{
    const double &mu = gprms.mu;
    const double &kappa = gprms.kappa;
    const double &tan_phi = gprms.DP_tan_phi;
    const double &DP_threshold_p = gprms.DP_threshold_p;

    //    const double &pmin = -gprms.IceTensileStrength;
    const double &pmax = gprms.IceCompressiveStrength;
    const double &qmax = gprms.IceShearStrength;

    if(p.p_tr < -DP_threshold_p || p.Jp_inv < 1)
    {
        // tear in tension or compress until original state
        double p_new = -DP_threshold_p;
        double Je_new = sqrt(-2.*p_new/kappa + 1.);
        Vector2d vSigma_new = Vector2d::Constant(1.)*sqrt(Je_new);  //Matrix2d::Identity() * pow(Je_new, 1./(double)d);
        p.Fe = p.U*vSigma_new.asDiagonal()*p.V.transpose();
        p.Jp_inv *= Je_new/p.Je_tr;
    }
    else
    {
        double q_n_1;

        if(p.p_tr > pmax)
        {
            q_n_1 = 0;
        }
        else
        {
            double q_from_dp = (p.p_tr+DP_threshold_p)*tan_phi;
            q_n_1 = min(q_from_dp,qmax);
            //            double q_from_failure_surface = 2*sqrt((pmax-p.p_tr)*(p.p_tr-pmin))*qmax/(pmax-pmin);
            //            q_n_1 = min(q_from_failure_surface, q_from_dp);
        }

        if(p.q_tr >= q_n_1)
        {
            // project onto YS
            double s_hat_n_1_norm = q_n_1/coeff1;
            //            Matrix2d B_hat_E_new = s_hat_n_1_norm*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() + Matrix2d::Identity()*(SigmaSquared.trace()/d);
            Vector2d vB_hat_E_new = s_hat_n_1_norm*(p.Je_tr/mu)*p.v_s_hat_tr.normalized() + Vector2d::Constant(1.)*(p.vSigmaSquared.sum()/d);
            Vector2d vSigma_new = vB_hat_E_new.array().sqrt().matrix();
            p.Fe = p.U*vSigma_new.asDiagonal()*p.V.transpose();
        }
    }
}


__device__ void GetParametersForGrain(short grain, double &pmin, double &pmax, double &qmax, double &beta, double &mSq)
{
//    double var1 = 1.0 + gprms.GrainVariability*0.05*(-10 + grain%21);
//    double var2 = 1.0 + gprms.GrainVariability*0.033*(-15 + (grain+3)%30);
    double var3 = 1.0 + gprms.GrainVariability*0.1*(-10 + (grain+4)%11);

    pmax = gprms.IceCompressiveStrength;// * var1;
    pmin = -gprms.IceTensileStrength;// * var2;
    qmax = gprms.IceShearStrength * var3;

    beta = gprms.NACC_beta;
//    beta = -pmin / pmax;
    //    double NACC_M = (2*qmax*sqrt(1+2*beta))/(pmax*(1+beta));
    //    mSq = NACC_M*NACC_M;
    mSq = (4*qmax*qmax*(1+2*beta))/((pmax*(1+beta))*(pmax*(1+beta)));
}


__device__ void CheckIfPointIsInsideFailureSurface(icy::Point &p)
{

    //    const double &beta = gprms.NACC_beta;
    //    const double &M_sq = gprms.NACC_Msq;
    //    const double &pmin = -gprms.IceTensileStrength;
    //    const double &pmax = gprms.IceCompressiveStrength;;
    //    const double &qmax = gprms.IceShearStrength;


    double beta, M_sq, pmin, pmax, qmax;
    GetParametersForGrain(p.grain, pmin, pmax, qmax, beta, M_sq);

    const double pmin2 = -3e6;
    if(p.p_tr<0)
    {
        if(p.p_tr<pmin2) {p.crushed = 1; return;}
        double q0 = 2*sqrt(-pmax*pmin)*qmax/(pmax-pmin);
        double k = -q0/pmin2;
        double q_limit = k*(p.p_tr-pmin2);
        if(p.q_tr > q_limit) {p.crushed = 1; return;}
    }
    else
    {
        double y = (1.+2.*beta)*p.q_tr*p.q_tr + M_sq*(p.p_tr + beta*pmax)*(p.p_tr - pmax);
        if(y > 0)
        {
            p.crushed = 1;
            p.utility_data |= status_crushed;
        }
    }
}




__global__ void partition_kernel_p2g(const unsigned gridX, const unsigned gridX_offset, const unsigned pitch_grid,
                                     const unsigned count_pts, const unsigned pitch_pts,
                                     const double *buffer_pts, double *buffer_grid)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= count_pts) return;

    const long long* ptr = reinterpret_cast<const long long*>(&buffer_pts[pitch_pts*icy::SimParams::idx_utility_data]);
    long long utility_data = ptr[pt_idx];
    if(utility_data & status_disabled) return; // point is disabled

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
            int idx_gridnode = j2 + (i2+halo*3)*gridY;  // two halo lines are reserved for the incoming halo data
            if(i2<(-halo) || j2<0 || i2>=(gridX+halo) || j2>=gridY) gpu_error_indicator = 1;

            // Udpate mass, velocity and force
            atomicAdd(&buffer_grid[0*pitch_grid + idx_gridnode], incM);
            atomicAdd(&buffer_grid[1*pitch_grid + idx_gridnode], incV[0]);
            atomicAdd(&buffer_grid[2*pitch_grid + idx_gridnode], incV[1]);
        }
}



void GPU_Partition::update_nodes()
{
    cudaSetDevice(Device);
    const int nGridNodes = prms->GridY * GridX_partition;
    int tpb = prms->tpb_Upd;
    int nBlocks = (nGridNodes + tpb - 1) / tpb;
    Eigen::Vector2d ind_center(prms->indenter_x, prms->indenter_y);

    partition_kernel_update_nodes<<<nBlocks, tpb, 0, streamCompute>>>(ind_center, nGridNodes, GridX_offset,
        nGridPitch, grid_array, indenter_force_accumulator);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("update_nodes");
}



__global__ void partition_kernel_update_nodes(const Eigen::Vector2d indCenter,
    const unsigned nNodes, const unsigned gridX_offset, const unsigned pitch_grid,
                                              double *_buffer_grid, double *indenter_force_accumulator)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nNodes) return;

    double *buffer_grid = _buffer_grid + 3*gprms.GridY*gprms.GridHaloSize;    // actual grid buffer comes after 3x halo regions
    double mass = buffer_grid[idx];
    if(mass == 0) return;

    const double &gravity = gprms.Gravity;
    const double &indRsq = gprms.IndRSq;
    const double &dt = gprms.InitialTimeStep;
    const double &ind_velocity = gprms.IndVelocity;
    const double &cellsize = gprms.cellsize;
    const double &vmax = gprms.vmax;
    const double &vmax_squared = gprms.vmax_squared;
    const unsigned &gridY = gprms.GridY;
    const unsigned &gridXTotal = gprms.GridXTotal;

    const Vector2d vco(ind_velocity,0);  // velocity of the collision object (indenter)

    Vector2d velocity(buffer_grid[1*pitch_grid + idx], buffer_grid[2*pitch_grid + idx]);
    velocity /= mass;
    velocity[1] -= gprms.dt_Gravity;
    if(velocity.squaredNorm() > vmax_squared) velocity = velocity.normalized()*vmax;

    Vector2i gi(idx/gridY + gridX_offset, idx%gridY);   // integer x-y index of the grid node
    Vector2d gnpos = gi.cast<double>()*cellsize;    // position of the grid node in the whole grid

    // indenter
    Vector2d n = gnpos - indCenter;
    if(n.squaredNorm() < indRsq)
    {
        // grid node is inside the indenter
        Vector2d vrel = velocity - vco;
        n.normalize();
        double vn = vrel.dot(n);   // normal component of the velocity
        if(vn < 0)
        {
            Vector2d vt = vrel - n*vn;   // tangential portion of relative velocity
            Vector2d prev_velocity = velocity;
            velocity = vco + vt;

            // force on the indenter
            Vector2d force = (prev_velocity-velocity)*mass/dt;
            float angle = atan2f((float)n[0],(float)n[1]);
            angle += icy::SimParams::pi;
            angle *= gprms.n_indenter_subdivisions/ (2*icy::SimParams::pi);
            int index = min(max((int)angle, 0), gprms.n_indenter_subdivisions-1);
            atomicAdd(&indenter_force_accumulator[0+2*index], force[0]);
            atomicAdd(&indenter_force_accumulator[1+2*index], force[1]);
        }
    }

    // attached bottom layer
    if(gi.y() <= 2) velocity.setZero();
    else if(gi.y() >= gridY-3 && velocity[1]>0) velocity[1] = 0;
    if(gi.x() <= 2 && velocity[0]<0) velocity[0] = 0;
    else if(gi.x() >= gridXTotal-3 && velocity[0]>0) velocity[0] = 0;

    // side boundary conditions
    //    int blocksGridX = gprms.BlockLength*gprms.cellsize_inv+5-2;
    //    int blocksGridY = gprms.BlockHeight/2*gprms.cellsize_inv+2;
    //    if(idx_x >= blocksGridX && idx_x <= blocksGridX + 2 && idx_y < blocksGridY) velocity.setZero();
    //    if(idx_x <= 7 && idx_x > 4 && idx_y < blocksGridY) velocity.setZero();

    // write the updated grid velocity back to memory
    buffer_grid[1*pitch_grid + idx] = velocity[0];
    buffer_grid[2*pitch_grid + idx] = velocity[1];
}




void GPU_Partition::receive_halos()
{
    cudaSetDevice(Device);
    const unsigned haloElementCount = prms->GridHaloSize*prms->GridY;
    const unsigned tpb = 512;   // threads per block
    const unsigned blocksPerGrid = (haloElementCount + tpb - 1) / tpb;
    partition_kernel_receive_halos<<<blocksPerGrid, tpb, 0, streamCompute>>>(haloElementCount, GridX_partition, nGridPitch, grid_array);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("receive_halos kernel execution");
}


__global__ void partition_kernel_receive_halos(const unsigned haloElementCount,
                                               const unsigned gridX, const unsigned pitch_grid, double *buffer_grid)
{
    const unsigned &halo = gprms.GridHaloSize;
    const unsigned &gridY = gprms.GridY;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= haloElementCount) return;
    for(int i=0; i<icy::SimParams::nGridArrays; i++)
    {
        // left halo
        buffer_grid[idx + i*pitch_grid + 3*halo*gridY] += buffer_grid[idx + i*pitch_grid + 0*halo*gridY];
        // right halo
        buffer_grid[idx + i*pitch_grid + (2*halo+gridX)*gridY] += buffer_grid[idx + i*pitch_grid + 1*halo*gridY];
    }
}




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
    host_side_utility_data = nullptr;

    pts_array = nullptr;
    grid_array = nullptr;
    indenter_force_accumulator = nullptr;
    vector_data_disabled_points = nullptr;
    for(int i=0;i<4;i++) point_transfer_buffer[i] = nullptr;
    device_side_utility_data = nullptr;
}

GPU_Partition::~GPU_Partition()
{
    cudaSetDevice(Device);
    cudaEventDestroy(eventCycleStart);
    cudaEventDestroy(eventCycleStop);
    cudaEventDestroy(event_grid_halo_sent);
    cudaEventDestroy(event_pts_sent);
    cudaEventDestroy(event_utility_data_transferred);

    cudaStreamDestroy(streamCompute);

    cudaFreeHost(host_side_indenter_force_accumulator);
    cudaFreeHost(host_side_utility_data);

    cudaFree(indenter_force_accumulator);
    cudaFree(pts_array);
    for(int i=0;i<4;i++) cudaFree(point_transfer_buffer[i]);
    cudaFree(vector_data_disabled_points);
    cudaFree(grid_array);
    cudaFree(device_side_utility_data);
    spdlog::info("Destructor invoked; partition {} on device {}", PartitionID, Device);
}

void GPU_Partition::initialize(int device, int partition)
{
    this->PartitionID = partition;
    this->Device = device;
    cudaSetDevice(Device);
    cudaEventCreate(&eventCycleStart);
    cudaEventCreate(&eventCycleStop);
    cudaEventCreate(&event_grid_halo_sent);
    cudaEventCreate(&event_pts_sent);
    cudaEventCreate(&event_utility_data_transferred);
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
    for(int i=0;i<4;i++)
    {
        err = cudaMalloc(&point_transfer_buffer[i], prms->VectorCapacity_transfer*icy::SimParams::nPtsArrays*sizeof(double));
        if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate");
    }

    // integer vector for disabled points
    err = cudaMalloc(&vector_data_disabled_points, prms->VectorCapacity_disabled*sizeof(unsigned));
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate vector_data_disabled_points");

    // grid
    size_t grid_size_local_requested = prms->GridY*(grid_x_capacity + 4*prms->GridHaloSize) * sizeof(double);
    err = cudaMallocPitch (&grid_array, &nGridPitch, grid_size_local_requested, icy::SimParams::nGridArrays);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate grid array");
    nGridPitch /= sizeof(double); // assume that this divides without remainder

    err = cudaMallocHost(&host_side_utility_data, utility_data_size*sizeof(unsigned));
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate host-side array");

    err = cudaMalloc(&device_side_utility_data, utility_data_size*sizeof(unsigned));
    if(err != cudaSuccess) throw std::runtime_error("GPU_Partition allocate space for utility data");

    spdlog::info("Partition {}-{}: allocated GridPitch {} ({}); Pts {}; Disabled {}; PtsTransfer {}; grid_size_local_requested {}",
                 PartitionID, Device, nGridPitch, nGridPitch/prms->GridY, nPtsPitch,
                 prms->VectorCapacity_disabled, prms->VectorCapacity_transfer, grid_size_local_requested);
}


void GPU_Partition::clear_utility_vectors()
{
    spdlog::info("P {} D {}, utility vectors clear",PartitionID,Device);
    cudaSetDevice(Device);
    cudaError_t err = cudaMemsetAsync(device_side_utility_data, 0, 3*sizeof(unsigned), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("initialize_utility_vectors");
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

    // also clear the point transfer vectors
    err = cudaMemsetAsync(point_transfer_buffer[0], 0, sizeof(unsigned), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("reset_grid point_tranfer_buffer");
    err = cudaMemsetAsync(point_transfer_buffer[1], 0, sizeof(unsigned), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("reset_grid point_tranfer_buffer");
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

