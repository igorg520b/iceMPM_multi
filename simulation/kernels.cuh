#include <cuda.h>
#include <cuda_runtime.h>
#include "point.h"



using namespace Eigen;

constexpr double d = 2; // dimensions
constexpr double coeff1 = 1.4142135623730950; // sqrt((6-d)/2.);
constexpr long long status_crushed = 0x10000;
constexpr long long status_disabled = 0x20000;

__device__ uint8_t gpu_error_indicator;
__constant__ icy::SimParams gprms;


__device__ Matrix2d KirchhoffStress_Wolper(const Matrix2d &F)
{
    const double &kappa = gprms.kappa;
    const double &mu = gprms.mu;

    // Kirchhoff stress as per Wolper (2019)
    double Je = F.determinant();
    Matrix2d b = F*F.transpose();
    Matrix2d PFt = mu*(1/Je)*dev(b) + kappa*0.5*(Je*Je-1.)*Matrix2d::Identity();
    return PFt;
}


__device__ Matrix2d Water(const double J)
{
    constexpr double gamma = 3;
    const double &kappa = gprms.kappa;

    Matrix2d PFt = kappa*( 1.-pow(J,-gamma))*Matrix2d::Identity();
    return PFt;
}

__global__ void partition_kernel_p2g(const int gridX, const int gridX_offset, const int pitch_grid,
                                     const int count_pts, const int pitch_pts,
                                     const double *buffer_pts, double *buffer_grid)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= count_pts) return;

    const long long* ptr = reinterpret_cast<const long long*>(&buffer_pts[pitch_pts*icy::SimParams::idx_utility_data]);
    long long utility_data = ptr[pt_idx];
    if(utility_data & status_disabled) return; // point is disabled

    //const double &dt = gprms.InitialTimeStep;
    //const double &vol = gprms.ParticleVolume;
    const double &h = gprms.cellsize;
    const double &h_inv = gprms.cellsize_inv;
    //const double &Dinv = gprms.Dp_inv;
    const double &particle_mass = gprms.ParticleMass;

    const int &gridY = gprms.GridY;
    //const int &gridXTotal = gprms.GridXTotal;
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

//    Matrix2d PFt = Water(buffer_pts[pt_idx + pitch_pts*icy::SimParams::idx_Jp_inv]);
    Matrix2d PFt = KirchhoffStress_Wolper(Fe);
    Matrix2d subterm2 = particle_mass*Bp - (gprms.dt_vol_Dpinv)*PFt;

    Eigen::Vector2i base_coord_i = (pos*h_inv - Vector2d::Constant(0.5)).cast<int>(); // coords of base grid node for point
    Vector2d base_coord = base_coord_i.cast<double>();
    Vector2d fx = pos*h_inv - base_coord;

    if(base_coord_i.x() - gridX_offset < (-halo)) gpu_error_indicator = 70;
    if(base_coord_i.y()<0) gpu_error_indicator = 71;
    if(base_coord_i.x() - gridX_offset>(gridX+halo-3)) gpu_error_indicator = 72;
    if(base_coord_i.y()>gridY-3) gpu_error_indicator = 73;

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
            int idx_gridnode = j2 + i2*gridY;
            // Udpate mass, velocity and force
            atomicAdd(&buffer_grid[0*pitch_grid + idx_gridnode + offset], incM);
            atomicAdd(&buffer_grid[1*pitch_grid + idx_gridnode + offset], incV[0]);
            atomicAdd(&buffer_grid[2*pitch_grid + idx_gridnode + offset], incV[1]);
        }
}


__global__ void partition_kernel_receive_halos_left(const int haloElementCount, const int gridX_partition,
                                               const int pitch_grid, double *buffer_grid,
                                               const double *halo0, const double *halo1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= haloElementCount) return;

    const int &gridY = gprms.GridY;
    for(int i=0; i<icy::SimParams::nGridArrays; i++)
    {
        buffer_grid[idx + i*pitch_grid] += halo0[idx + i*pitch_grid];
//        buffer_grid[idx + i*pitch_grid + gridY*gridX_partition] += halo1[idx + i*pitch_grid];
    }
}

__global__ void partition_kernel_receive_halos_right(const int haloElementCount, const int gridX_partition,
                                               const int pitch_grid, double *buffer_grid,
                                               const double *halo0, const double *halo1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= haloElementCount) return;

    const int &gridY = gprms.GridY;
    for(int i=0; i<icy::SimParams::nGridArrays; i++)
    {
//        buffer_grid[idx + i*pitch_grid] += halo0[idx + i*pitch_grid];
        buffer_grid[idx + i*pitch_grid + gridY*gridX_partition] += halo1[idx + i*pitch_grid];
    }
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

    //const double &gravity = gprms.Gravity;
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
//    if(gi.y() <= 2) velocity[1] = 0;
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

//    p.Jp_inv *= (Matrix2d::Identity() + dt*p.Bp).determinant();  // for water model

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


    // only tranfer the points if this feature is enabled this particular step
    constexpr int fly_threshold = 3;
    if(enablePointTransfer)
    {
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
        double sqrt_Je_new = sqrt(Je_new);

        Vector2d vSigma_new(sqrt_Je_new,sqrt_Je_new); //= Vector2d::Constant(1.)*sqrt(Je_new);  //Matrix2d::Identity() * pow(Je_new, 1./(double)d);
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
//    double NACC_M = (2*qmax*sqrt(1+2*beta))/(pmax-pmin);
//    mSq = NACC_M*NACC_M;
    mSq = (4*qmax*qmax*(1+2*beta))/((pmax-pmin)*(pmax-pmin));
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





// deviatoric part of a diagonal matrix
__device__ Vector2d dev_d(Vector2d Adiag)
{
    return Adiag - Adiag.sum()/2*Vector2d::Constant(1.);
}

__device__ Eigen::Matrix2d dev(Eigen::Matrix2d A)
{
    return A - A.trace()/2*Eigen::Matrix2d::Identity();
}

