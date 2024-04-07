#ifndef P_SIM_H
#define P_SIM_H

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include <Eigen/Core>
#include "rapidjson/reader.h"
#include "rapidjson/document.h"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

// variables related to the formulation of the model

namespace icy { struct SimParams; }

struct icy::SimParams
{
public:
    constexpr static double pi = 3.14159265358979323846;
    constexpr static unsigned dim = 2;
    constexpr static unsigned nGridArrays = 3;

    // index of the corresponding array in SoA
    constexpr static size_t idx_utility_data = 0;
    constexpr static size_t idx_P = idx_utility_data + 1;
    constexpr static size_t idx_Q = idx_P + 1;
    constexpr static size_t idx_Jp_inv = idx_Q + 1;
    constexpr static size_t posx = idx_Jp_inv + 1;
    constexpr static size_t velx = posx + 2;
    constexpr static size_t Fe00 = velx + 2;
    constexpr static size_t Bp00 = Fe00 + 4;
    constexpr static size_t nPtsArrays = Bp00 + 4;

    unsigned n_indenter_subdivisions;
    unsigned tpb_P2G, tpb_Upd, tpb_G2P;  // threads per block for each operation

    unsigned nPtsTotal;
    unsigned GridXTotal, GridY, GridTotal;
    double GridXDimension;

    double InitialTimeStep, SimulationEndTime;
    unsigned UpdateEveryNthStep; // run N steps without update
    unsigned SimulationStep;
    double SimulationTime;

    // material properties
    double Gravity, Density, PoissonsRatio, YoungsModulus;
    double lambda, mu, kappa; // Lame

    double IceCompressiveStrength, IceTensileStrength, IceShearStrength;
    double NACC_beta, NACC_M, NACC_Msq;     // these are all computed

    double DP_tan_phi, DP_threshold_p;

    // indentation params
    double IndDiameter, IndRSq, IndVelocity, IndDepth;

    double cellsize, cellsize_inv, Dp_inv;
    double xmin, xmax, ymin, ymax;            // bounding box of the material
    unsigned nxmin, nxmax, nymin, nymax;         // same, but nuber of grid cells

    double ParticleVolume, ParticleMass, ParticleViewSize;

    double indenter_x, indenter_x_initial, indenter_y, indenter_y_initial;
    double Volume;  // total volume (area) of the object
    unsigned SetupType;  // 0 - ice block horizontal indentation; 1 - cone uniaxial compression
    double GrainVariability;

    // multi-GPU params
    int GridHaloSize;  // number of grid slices (perpendicular to the x-axis) for "halo" transfers
    double ExtraSpaceForIncomingPoints;     // percentage of points per partition
    double PointsTransferBufferFraction;    // space for points that can "fly over" per simulation step

    unsigned nPartitions; // number of partitions (ideally, one partition per device)


    // computed parameters/properties
    double dt_vol_Dpinv, dt_Gravity, vmax, vmax_squared;


    void Reset();
    std::string ParseFile(std::string fileName);

    void ComputeLame();
    void ComputeCamClayParams2();
    void ComputeHelperVariables();
    void ComputeIntegerBlockCoords();
    double PointsPerCell() {return nPtsTotal/(Volume/(cellsize*cellsize));}
    int AnimationFrameNumber() { return SimulationStep / UpdateEveryNthStep;}
    size_t IndenterArraySize() { return sizeof(double)*n_indenter_subdivisions*dim; }

    // grid cell from point's coordinates
    int CellIdx(float x) { return (int)(x*cellsize_inv+0.5); }
    int PointCellIndex(float x, float y) { return CellIdx(x)*GridY + CellIdx(y);}

};

#endif
