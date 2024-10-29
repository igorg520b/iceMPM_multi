#include "parameters_sim.h"
#include <spdlog/spdlog.h>


void icy::SimParams::Reset()
{
    nPtsTotal = 0;
    n_indenter_subdivisions = 100;

    InitialTimeStep = 3.e-5;
    YoungsModulus = 5.e8;
    GridXTotal = 128;
    GridY = 55;
    ParticleViewSize = 2.5f;
    GridXDimension = 3.33;

    SimulationEndTime = 12;

    PoissonsRatio = 0.3;
    Gravity = 9.81;
    Density = 916;

    IndDiameter = 0.324;
    IndVelocity = 0.2;
    IndDepth = 0.25;//0.101;

    SimulationStep = 0;
    SimulationTime = 0;

    IceCompressiveStrength = 100e6;
    IceTensileStrength = 10e6;
    IceShearStrength = 6e6;
    IceTensileStrength2 = 5e6;

    DP_tan_phi = std::tan(30*pi/180.);
    DP_threshold_p = 1e3;

    tpb_P2G = 256;
    tpb_Upd = 512;
    tpb_G2P = 128;

    indenter_x = indenter_x_initial = indenter_y = indenter_y_initial = 0;
    SetupType = 0;
    GrainVariability = 0.50;

    GridHaloSize = 5;
    ExtraSpaceForIncomingPoints = 0.2;  // 20%
    PointsTransferBufferFraction = 0.05; // % of points that could "fly over" during a given cycle
    nPartitions = 3;        // one partition of single-gpu; >1 for multi-gpu

    RebalanceThresholdFreeSpaceRemaining = 0.15;
    RebalanceThresholdDisabledPercentage = 0.01;
    PointTransferFrequency = 2;

    ComputeLame();
    ComputeCamClayParams2();
    ComputeHelperVariables();
    spdlog::info("SimParams reset");
}


std::string icy::SimParams::ParseFile(std::string fileName)
{
    spdlog::info("SimParams ParseFile {}",fileName);
    if(!std::filesystem::exists(fileName)) throw std::runtime_error("configuration file is not found");
    std::ifstream fileStream(fileName);
    std::string strConfigFile;
    strConfigFile.resize(std::filesystem::file_size(fileName));
    fileStream.read(strConfigFile.data(), strConfigFile.length());
    fileStream.close();

    rapidjson::Document doc;
    doc.Parse(strConfigFile.data());
    if(!doc.IsObject()) throw std::runtime_error("configuration file is not JSON");

    if(doc.HasMember("InitialTimeStep")) InitialTimeStep = doc["InitialTimeStep"].GetDouble();
    if(doc.HasMember("YoungsModulus")) YoungsModulus = doc["YoungsModulus"].GetDouble();
    if(doc.HasMember("GridX")) GridXTotal = doc["GridX"].GetInt();
    if(doc.HasMember("GridY")) GridY = doc["GridY"].GetInt();
    if(doc.HasMember("GridXDimension")) GridXDimension = doc["GridXDimension"].GetDouble();
    if(doc.HasMember("ParticleViewSize")) ParticleViewSize = doc["ParticleViewSize"].GetDouble();
    if(doc.HasMember("SimulationEndTime")) SimulationEndTime = doc["SimulationEndTime"].GetDouble();
    if(doc.HasMember("PoissonsRatio")) PoissonsRatio = doc["PoissonsRatio"].GetDouble();
    if(doc.HasMember("Gravity")) Gravity = doc["Gravity"].GetDouble();
    if(doc.HasMember("Density")) Density = doc["Density"].GetDouble();
    if(doc.HasMember("IndDiameter")) IndDiameter = doc["IndDiameter"].GetDouble();
    if(doc.HasMember("IndVelocity")) IndVelocity = doc["IndVelocity"].GetDouble();
    if(doc.HasMember("IndDepth")) IndDepth = doc["IndDepth"].GetDouble();

    if(doc.HasMember("IceCompressiveStrength")) IceCompressiveStrength = doc["IceCompressiveStrength"].GetDouble();
    if(doc.HasMember("IceTensileStrength")) IceTensileStrength = doc["IceTensileStrength"].GetDouble();
    if(doc.HasMember("IceTensileStrength2")) IceTensileStrength2 = doc["IceTensileStrength2"].GetDouble();
    if(doc.HasMember("IceShearStrength")) IceShearStrength = doc["IceShearStrength"].GetDouble();

    if(doc.HasMember("DP_phi")) DP_tan_phi = std::tan(doc["DP_phi"].GetDouble()*pi/180);
    if(doc.HasMember("DP_threshold_p")) DP_threshold_p = doc["DP_threshold_p"].GetDouble();
    if(doc.HasMember("GrainVariability")) GrainVariability = doc["GrainVariability"].GetDouble();

    if(doc.HasMember("tpb_P2G")) tpb_P2G = doc["tpb_P2G"].GetInt();
    if(doc.HasMember("tpb_Upd")) tpb_Upd = doc["tpb_Upd"].GetInt();
    if(doc.HasMember("tpb_G2P")) tpb_G2P = doc["tpb_G2P"].GetInt();
    if(doc.HasMember("GridHaloSize")) GridHaloSize = doc["GridHaloSize"].GetInt();
    if(doc.HasMember("nPartitions")) nPartitions = doc["nPartitions"].GetInt();

    if(doc.HasMember("ExtraSpaceForIncomingPoints")) ExtraSpaceForIncomingPoints = doc["ExtraSpaceForIncomingPoints"].GetDouble();


    ComputeCamClayParams2();
    ComputeHelperVariables();

    std::cout << "loaded parameters file " << fileName << std::endl;
    std::cout << "GridXDimension " << GridXDimension << std::endl;
    std::cout << "cellsize " << cellsize << std::endl;

    if(!doc.HasMember("InputRawPoints"))
    {
        spdlog::critical("InputRawPoints entry is missing in JSON config file");
        throw std::runtime_error("config parameter missing");
    }

    std::string result = doc["InputRawPoints"].GetString();
    spdlog::info("ParseFile; raw point data {}",result);
    return result;
}

void icy::SimParams::ComputeLame()
{
    lambda = YoungsModulus*PoissonsRatio/((1+PoissonsRatio)*(1-2*PoissonsRatio));
    mu = YoungsModulus/(2*(1+PoissonsRatio));
    kappa = mu*2./3. + lambda;
}

void icy::SimParams::ComputeHelperVariables()
{
    UpdateEveryNthStep = (int)(1.f/(200*InitialTimeStep));
    cellsize = GridXDimension/GridXTotal;
    cellsize_inv = 1./cellsize;
    Dp_inv = 4./(cellsize*cellsize);
    IndRSq = IndDiameter*IndDiameter/4.;
    dt_vol_Dpinv = InitialTimeStep*ParticleVolume*Dp_inv;
    dt_Gravity = InitialTimeStep*Gravity;
    vmax = 0.5*cellsize/InitialTimeStep;
    vmax_squared = vmax*vmax;

    VectorCapacity_transfer = nPtsTotal/nPartitions * PointsTransferBufferFraction;

    gbOffset = GridY * GridHaloSize;
}

void icy::SimParams::ComputeCamClayParams2()
{
    ComputeLame();
    NACC_beta = IceTensileStrength/IceCompressiveStrength;
    const double &beta = NACC_beta;
    const double &q = IceShearStrength;
    const double &p0 = IceCompressiveStrength;
    NACC_M = (2*q*sqrt(1+2*beta))/(p0*(1+beta));
    NACC_Msq = NACC_M*NACC_M;
    spdlog::info("ComputeCamClayParams2() done");
}

void icy::SimParams::ComputeIntegerBlockCoords()
{
    nxmin = floor(xmin/cellsize);
    nxmax = ceil(xmax/cellsize);
    nymin = floor(ymin/cellsize);
    nymax = ceil(ymax/cellsize);
}
