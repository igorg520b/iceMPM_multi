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


// =========================================  KERNELS




__global__ void partition_kernel_p2g(const int gridX, const int gridX_offset, const int pitch_grid,
                                     const int count_pts, const int pitch_pts,
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

    const int &gridY = gprms.GridY;
    const int &gridXTotal = gprms.GridXTotal;
    const int &halo = gprms.GridHaloSize;

    const int &offset = gprms.gbOffset;

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
//    Matrix2d subterm2 = particle_mass*Bp - (gprms.InitialTimeStep*gprms.ParticleVolume*gprms.Dp_inv)*PFt;

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
            int i2 = i+base_coord_i[0] - gridX_offset;
            int j2 = j+base_coord_i[1];
            if(i2<(-halo)) gpu_error_indicator = 70;
            if(j2<0) gpu_error_indicator = 71;
            if(i2>(gridX+halo-3)) gpu_error_indicator = 72;
            if(j2>gridY-3) gpu_error_indicator = 73;

            int idx_gridnode = j2 + i2*gridY;
            // Udpate mass, velocity and force
            atomicAdd(&buffer_grid[0*pitch_grid + idx_gridnode + offset], incM);
            atomicAdd(&buffer_grid[1*pitch_grid + idx_gridnode + offset], incV[0]);
            atomicAdd(&buffer_grid[2*pitch_grid + idx_gridnode + offset], incV[1]);
        }
}


__global__ void partition_kernel_receive_halos(const int haloElementCount,
                                               const int gridX, const int gridX_offset,
                                               const int pitch_grid, double *buffer_grid,
                                               const double *halo0, const double *halo1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= haloElementCount) return;

    const int &halo = gprms.GridHaloSize;
    const int &gridY = gprms.GridY;
    const int offset = gprms.gbOffset;

    for(int i=0; i<icy::SimParams::nGridArrays; i++)
    {
        buffer_grid[idx + i*pitch_grid] += halo0[idx + i*pitch_grid];
        buffer_grid[idx + i*pitch_grid + gridY*gridX] += halo1[idx + i*pitch_grid];
    }
}

void GPU_Partition::receive_halos()
{
    cudaSetDevice(Device);
    const int haloElementCount = prms->GridHaloSize*prms->GridY*2;
    const int tpb = prms->tpb_Upd;   // threads per block
    const int blocksPerGrid = (haloElementCount + tpb - 1) / tpb;
    partition_kernel_receive_halos<<<blocksPerGrid, tpb, 0, streamCompute>>>(haloElementCount, GridX_partition, GridX_offset,
                                                                             nGridPitch, grid_array,
                                                                             halo_transfer_buffer[0], halo_transfer_buffer[1]);

    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("receive_halos kernel execution");
}


__global__ void partition_kernel_update_nodes(const Eigen::Vector2d indCenter,
                                              const int nNodes, const int gridX_offset, const int pitch_grid,
                                              double *buffer_grid, double *indenter_force_accumulator)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nNodes) return;

    const int &halo = gprms.GridHaloSize;
    const int &gridY = gprms.GridY;

    double mass = buffer_grid[idx];
    double vx = buffer_grid[1*pitch_grid + idx];
    double vy = buffer_grid[2*pitch_grid + idx];
    if(mass == 0) return;

    const double &gravity = gprms.Gravity;
    const double &indRsq = gprms.IndRSq;
    const double &dt = gprms.InitialTimeStep;
    const double &ind_velocity = gprms.IndVelocity;
    const double &cellsize = gprms.cellsize;
    const double &vmax = gprms.vmax;
    const double &vmax_squared = gprms.vmax_squared;
    const int &gridXTotal = gprms.GridXTotal;

    const Vector2d vco(ind_velocity,0);  // velocity of the collision object (indenter)

    Vector2i gi(idx/gridY+gridX_offset-halo, idx%gridY);   // integer x-y index of the grid node
    Vector2d velocity(vx, vy);
    velocity /= mass;
    velocity[1] -= gprms.dt_Gravity;
    if(velocity.squaredNorm() > vmax_squared) velocity = velocity.normalized()*vmax;

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
    else if(gi.y() >= gridY-4 && velocity[1]>0) velocity[1] = 0;
    if(gi.x() <= 2 && velocity[0]<0) velocity[0] = 0;
    else if(gi.x() >= gridXTotal-4 && velocity[0]>0) velocity[0] = 0;

    // side boundary conditions would go here

    // write the updated grid velocity back to memory
    buffer_grid[1*pitch_grid + idx] = velocity[0];
    buffer_grid[2*pitch_grid + idx] = velocity[1];
}



__global__ void partition_kernel_g2p(const bool recordPQ, const bool enablePointTransfer,
                                     const int gridX, const int gridX_offset, const int pitch_grid,
                                     const int count_pts, const int pitch_pts,
                                     double *buffer_pts, const double *buffer_grid,
                                     int *utility_data,
                                     const int VectorCapacity_transfer,
                                     double *point_buffer_left, double *point_buffer_right)
{
    const int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= count_pts) return;

    // skip if a point is disabled
    icy::Point p;
    long long* ptr = reinterpret_cast<long long*>(&buffer_pts[pt_idx + pitch_pts*icy::SimParams::idx_utility_data]);
    p.utility_data = *ptr;
    if(p.utility_data & status_disabled) return; // point is disabled

    const int &halo = gprms.GridHaloSize;
    const double &h_inv = gprms.cellsize_inv;
    const double &dt = gprms.InitialTimeStep;
    const int &gridY = gprms.GridY;
    const double &mu = gprms.mu;
    const double &kappa = gprms.kappa;
    const int &offset = gprms.gbOffset;

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

    // coords of base grid node for point
    Eigen::Vector2i base_coord_i = (p.pos*h_inv - Vector2d::Constant(0.5)).cast<int>();
    Vector2d base_coord = base_coord_i.cast<double>();
    Vector2d fx = p.pos*h_inv - base_coord;

    // optimized method of computing the quadratic weight function without conditional operators
    Array2d arr_v0 = 1.5 - fx.array();
    Array2d arr_v1 = fx.array() - 1.0;
    Array2d arr_v2 = fx.array() - 0.5;
    Array2d ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    p.velocity.setZero();
    p.Bp.setZero();

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            Vector2d dpos = Vector2d(i, j) - fx;
            double weight = ww[i][0]*ww[j][1];

            int i2 = i+base_coord_i[0]-gridX_offset;
            int j2 = j+base_coord_i[1];
            int idx_gridnode = j2 + i2*gridY;

            Vector2d node_velocity;
            node_velocity[0] = buffer_grid[1*pitch_grid + idx_gridnode + offset];
            node_velocity[1] = buffer_grid[2*pitch_grid + idx_gridnode + offset];
            p.velocity += weight * node_velocity;
            p.Bp += (4.*h_inv)*weight *(node_velocity*dpos.transpose());
        }

    // Advection and update of the deformation gradient
    p.pos += p.velocity * dt;
    p.Fe = (Matrix2d::Identity() + dt*p.Bp) * p.Fe;     // p.Bp is the gradient of the velocity vector (it seems)

    ComputePQ(p, kappa, mu);    // pre-computes USV, p, q, etc.

    if(!(p.utility_data & status_crushed)) CheckIfPointIsInsideFailureSurface(p);
    if(p.utility_data & status_crushed) Wolper_Drucker_Prager(p);

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
    *ptr = p.utility_data; // includes crushed/disable status and grain number

    // at the end of each cycle, PQ are recorded for visualization
    if(recordPQ)
    {
        buffer_pts[pt_idx + pitch_pts*icy::SimParams::idx_P] = p.p_tr;
        buffer_pts[pt_idx + pitch_pts*icy::SimParams::idx_Q] = p.q_tr;
    }

    // check if a points needs to be transferred to adjacent partition
    int base_coord_x = (int)(p.pos.x()*h_inv - 0.5); // updated after the point has moved


    const int keep_track_threshold = halo/2-1;
    if((base_coord_x - gridX_offset) < -keep_track_threshold)
    {
        int deviation = -(base_coord_x - gridX_offset);
        atomicMax(&utility_data[GPU_Partition::idx_pts_max_extent], deviation);
    }
    else if(base_coord_x - (gridX_offset+gridX-3) > keep_track_threshold)
    {
        int deviation = base_coord_x - (gridX_offset+gridX-3);
        atomicMax(&utility_data[GPU_Partition::idx_pts_max_extent], deviation);
    }

    // only tranfer the points if this feature is enabled this particular step
    constexpr int fly_threshold = 3;
    if(enablePointTransfer)
    {
        if((base_coord_x - gridX_offset) < -fly_threshold)
        {
            // point transfers to the left
            int fly_idx = atomicAdd(&utility_data[GPU_Partition::idx_transfer_to_left], 1);  // reserve buffer index
            if(fly_idx < VectorCapacity_transfer)
            {
                // only perform this procedure if there is space in the buffer
                PreparePointForTransfer(pt_idx, fly_idx, point_buffer_left, pitch_pts, buffer_pts);
                *ptr = status_disabled; // includes crushed/disable status and grain number
            }
            else
                utility_data[GPU_Partition::idx_transfer_to_left] = VectorCapacity_transfer;
        }
        else if(base_coord_x - (gridX_offset+gridX-3) > fly_threshold)
        {
            // point transfers to the right
            int fly_idx = atomicAdd(&utility_data[GPU_Partition::idx_transfer_to_right], 1);  // reserve buffer index
            if(fly_idx < VectorCapacity_transfer)
            {
                PreparePointForTransfer(pt_idx, fly_idx, point_buffer_right, pitch_pts, buffer_pts);
                *ptr = status_disabled; // includes crushed/disable status and grain number
            }
            else
                utility_data[GPU_Partition::idx_transfer_to_right] = VectorCapacity_transfer;
        }
    }
}

__device__ void PreparePointForTransfer(const int pt_idx, const int index_in_transfer_buffer,
                                        double *point_transfer_buffer, const int pitch_pts,
                                        const double *buffer_pts)
{
    // check buffer boundary
    for(int i=0;i<icy::SimParams::nPtsArrays;i++)
        point_transfer_buffer[i + index_in_transfer_buffer*icy::SimParams::nPtsArrays] = buffer_pts[pt_idx + pitch_pts*i];
}


__global__ void partition_kernel_receive_points(const int count_transfer,
                                                const int count_pts, const int pitch_pts,
                                                double *buffer_pts,
                                                double *transfer_buffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count_transfer) return;

    int idx_in_soa = count_pts + idx;
    if(idx_in_soa >= pitch_pts) { gpu_error_indicator = 5; return; } // no space for incoming points

    // copy point data
    for(int i=0;i<icy::SimParams::nPtsArrays;i++)
    {
        buffer_pts[idx_in_soa + i*pitch_pts] = transfer_buffer[i + icy::SimParams::nPtsArrays*idx];
    }
}





// =========================================  DEVICE FUNCTIONS




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
            //Matrix2d B_hat_E_new = s_hat_n_1_norm*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() + Matrix2d::Identity()*(SigmaSquared.trace()/d);
            Vector2d vB_hat_E_new = s_hat_n_1_norm*(p.Je_tr/mu)*p.v_s_hat_tr.normalized() +
                                    Vector2d::Constant(1.)*(p.vSigmaSquared.sum()/d);
            Vector2d vSigma_new = vB_hat_E_new.array().sqrt().matrix();
            p.Fe = p.U*vSigma_new.asDiagonal()*p.V.transpose();
        }
    }

}


__device__ void GetParametersForGrain(short grain, double &pmin, double &pmax, double &qmax, double &beta, double &mSq, double &pmin2)
{
    //    double var1 = 1.0 + gprms.GrainVariability*0.05*(-10 + grain%21);
    double var2 = 1.0 + gprms.GrainVariability*0.033*(-15 + (grain+3)%30);
    double var3 = 1.0 + gprms.GrainVariability*0.1*(-10 + (grain+4)%11);

    pmax = gprms.IceCompressiveStrength;// * var1;
    pmin = -gprms.IceTensileStrength;// * var2;
    qmax = gprms.IceShearStrength * var3;
    pmin2 = -gprms.IceTensileStrength2 * var2;

    beta = gprms.NACC_beta;
//    beta = -pmin / pmax;
    double NACC_M = (2*qmax*sqrt(1+2*beta))/(pmax*(1+beta));
    mSq = NACC_M*NACC_M;
    mSq = (4*qmax*qmax*(1+2*beta))/((pmax*(1+beta))*(pmax*(1+beta)));
}


__device__ void CheckIfPointIsInsideFailureSurface(icy::Point &p)
{
    double beta, M_sq, pmin, pmax, qmax, pmin2;
    GetParametersForGrain(p.grain, pmin, pmax, qmax, beta, M_sq, pmin2);

    if(p.p_tr<0)
    {
        if(p.p_tr<pmin2) {p.utility_data |= status_crushed; return;}
        double q0 = 2*sqrt(-pmax*pmin)*qmax/(pmax-pmin);
        double k = -q0/pmin2;
        double q_limit = k*(p.p_tr-pmin2);
        if(p.q_tr > q_limit) {p.utility_data |= status_crushed; return;}
    }
    else
    {
        double y = (1.+2.*beta)*p.q_tr*p.q_tr + M_sq*(p.p_tr + beta*pmax)*(p.p_tr - pmax);
        if(y > 0)
        {
            p.utility_data |= status_crushed;
        }
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
    cudaMallocPitch (&grid_array, &nGridPitch, grid_size_local_requested, icy::SimParams::nGridArrays);
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


// ============================== main simulation steps
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

void GPU_Partition::update_nodes()
{
    cudaSetDevice(Device);
    const int nGridNodes = prms->GridY * (GridX_partition + 2*prms->GridHaloSize);

    int tpb = prms->tpb_Upd;
    int nBlocks = (nGridNodes + tpb - 1) / tpb;
    Eigen::Vector2d ind_center(prms->indenter_x, prms->indenter_y);

    partition_kernel_update_nodes<<<nBlocks, tpb, 0, streamCompute>>>(ind_center, nGridNodes, GridX_offset,
                                                                      nGridPitch, grid_array, indenter_force_accumulator);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("update_nodes");
}




// GRID HALO


double* GPU_Partition::getHaloAddress(int whichHalo, int whichGridArray)
{
    if(whichHalo == 0)
    {
        // left halo
        return grid_array + (prms->GridY * prms->GridHaloSize*4) + whichGridArray*nGridPitch;
    }
    else if(whichHalo == 1)
    {
        // right halo
        return grid_array + prms->GridY * (GridX_partition + 4*prms->GridHaloSize) + whichGridArray*nGridPitch;
    }
    else throw std::runtime_error("getHaloAddress");
}

double* GPU_Partition::getHaloReceiveAddress(int whichHalo, int whichGridArray)
{
    return grid_array + (prms->GridY * prms->GridHaloSize*whichHalo*2) + whichGridArray*nGridPitch;
}





// =====================================

void GPU_Partition::g2p(const bool recordPQ, const bool enablePointTransfer)
{
    cudaError_t err;
    err = cudaSetDevice(Device);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("g2p cudaSetDevice");

    // clear the counters for (0) left transfer, (1) right transfer, (2) added to the end of the list
    err = cudaMemsetAsync(device_side_utility_data, 0, utility_data_size*sizeof(int), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("g2p cudaMemset");

    const int &n = nPts_partition;
    const int &tpb = prms->tpb_G2P;
    const int nBlocks = (n + tpb - 1) / tpb;

    int gridxoffset = 0; // todo: change to GridX_offset
    int gridxpartition = prms->GridXTotal; // todo change to GridX_partition;

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

