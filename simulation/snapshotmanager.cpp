#include "snapshotmanager.h"
#include "model.h"

#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <utility>


void icy::SnapshotManager::SortPoints(std::vector<std::tuple<float,float,short>> &buffer)
{
    spdlog::info("sort start");
    std::sort(buffer.begin(),buffer.end(),[this](const std::tuple<float,float,short> &a, std::tuple<float,float,short> &b){
        float ax = std::get<0>(a);
        float ay = std::get<1>(a);
        float bx = std::get<0>(b);
        float by = std::get<1>(b);
        int aidx = model->prms.PointCellIndex(ax,ay);
        int bidx = model->prms.PointCellIndex(bx,by);
        return aidx<bidx;});
    spdlog::info("sort done");
}

void icy::SnapshotManager::LoadRawPoints(std::string fileName)
{
    spdlog::info("reading raw points file {}",fileName);
    if(!std::filesystem::exists(fileName)) throw std::runtime_error("error reading raw points file - no file");;

    H5::H5File file(fileName, H5F_ACC_RDONLY);

    H5::DataSet dataset_grains = file.openDataSet("llGrainIDs");
    hsize_t nPoints;
    dataset_grains.getSpace().getSimpleExtentDims(&nPoints, NULL);
    model->prms.nPtsTotal = nPoints;
    model->prms.ComputeHelperVariables();

    // allocate space host-side
    model->gpu.hssoa.Allocate(nPoints*(1+model->prms.ExtraSpaceForIncomingPoints));
    model->gpu.hssoa.size = nPoints;

    // read
    file.openDataSet("x").read(model->gpu.hssoa.getPointerToPosX(), H5::PredType::NATIVE_DOUBLE);
    file.openDataSet("y").read(model->gpu.hssoa.getPointerToPosY(), H5::PredType::NATIVE_DOUBLE);
    dataset_grains.read(model->gpu.hssoa.host_buffer, H5::PredType::NATIVE_UINT64);

    // read volume attribute
    H5::Attribute att_volume = dataset_grains.openAttribute("volume");
    float volume;
    att_volume.read(H5::PredType::NATIVE_FLOAT, &volume);
    model->prms.Volume = (double)volume;
    file.close();

    // get block dimensions
    std::pair<Eigen::Vector2d, Eigen::Vector2d> boundaries = model->gpu.hssoa.getBlockDimensions();
    model->prms.xmin = boundaries.first.x();
    model->prms.ymin = boundaries.first.y();
    model->prms.xmax = boundaries.second.x();
    model->prms.ymax = boundaries.second.y();


    const double &h = model->prms.cellsize;
    const double box_x = model->prms.GridXTotal*h;
    const double length = model->prms.xmax - model->prms.xmin;
    const double x_offset = (box_x - length)/2;
    const double y_offset = 2*h;

    Eigen::Vector2d offset(x_offset, y_offset);
    model->gpu.hssoa.offsetBlock(offset);
    model->gpu.hssoa.RemoveDisabledAndSort(model->prms.cellsize_inv, model->prms.GridY);
    model->gpu.hssoa.InitializeBlock();

    // set indenter starting position
    const double block_left = x_offset;
    const double block_top = model->prms.ymax + y_offset;

    const double r = model->prms.IndDiameter/2;
    const double ht = r - model->prms.IndDepth;
    const double x_ind_offset = sqrt(r*r - ht*ht);

    model->prms.indenter_x = floor((block_left-x_ind_offset)/h)*h;
    if(model->prms.SetupType == 0)
        model->prms.indenter_y = block_top + ht;
    else if(model->prms.SetupType == 1)
        model->prms.indenter_y = ceil(block_top/h)*h;

    model->prms.indenter_x_initial = model->prms.indenter_x;
    model->prms.indenter_y_initial = model->prms.indenter_y;

    // particle volume and mass
    model->prms.ParticleVolume = model->prms.Volume/nPoints;
    model->prms.ParticleMass = model->prms.ParticleVolume * model->prms.Density;

    // allocate GPU partitions
    model->gpu.device_allocate_arrays();

    // transfer points to device(s)
    model->gpu.transfer_ponts_to_device();


    model->Reset();
    model->Prepare();

    spdlog::info("LoadRawPoints done; nPoitns {}",nPoints);
}



void icy::SnapshotManager::SavePQ(std::string outputDirectory)
{
    /*
    std::filesystem::path odp(outputDirectory);
    if(!std::filesystem::is_directory(odp) || !std::filesystem::exists(odp)) std::filesystem::create_directory(odp);
    std::filesystem::path odp2(outputDirectory+"/"+directory_pq);
    if(!std::filesystem::is_directory(odp2) || !std::filesystem::exists(odp2)) std::filesystem::create_directory(odp2);

    const int current_frame_number = model->prms.AnimationFrameNumber();
    char fileName[20];
    snprintf(fileName, sizeof(fileName), "pq%05d.h5", current_frame_number);
    std::string filePath = outputDirectory + "/" + directory_pq + "/" + fileName;
    spdlog::info("saving NC frame {} to file {}", current_frame_number, filePath);


    const int &n = model->prms.nPtsTotal;
    buffer1.clear();
    buffer1.reserve(n*2);
    buffer2.clear();
    buffer2.reserve(n*2);

    for(int i=0;i<n;i++)
    {
        auto [p,q] = icy::Point::getPQ(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch,i);
        uint8_t crushed = icy::Point::getCrushedStatus(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch,i);
        if(crushed) {buffer1.push_back(p);buffer1.push_back(q);}
        else {buffer2.push_back(p);buffer2.push_back(q);}
    }

    H5::H5File file(filePath, H5F_ACC_TRUNC);

    hsize_t dims_pq_intact[2] = {buffer2.size()/2, 2};
    H5::DataSpace dataspace_pq_intact(2, dims_pq_intact);
    H5::DataSet dataset_pq_intact = file.createDataSet("PQ_intact", H5::PredType::NATIVE_DOUBLE, dataspace_pq_intact);
    dataset_pq_intact.write(buffer2.data(), H5::PredType::NATIVE_DOUBLE);
    SaveParametersAsAttributes(dataset_pq_intact);

    hsize_t dims_pq_crushed[2] = {buffer1.size()/2, 2};
    H5::DataSpace dataspace_pq_crushed(2, dims_pq_crushed);
    H5::DataSet dataset_pq_crushed = file.createDataSet("PQ_crushed", H5::PredType::NATIVE_DOUBLE, dataspace_pq_crushed);

    dataset_pq_crushed.write(buffer1.data(), H5::PredType::NATIVE_DOUBLE);
    file.close();
*/
}



void icy::SnapshotManager::SaveSnapshot(std::string outputDirectory)
{
    /*
    std::filesystem::path odp(outputDirectory);
    if(!std::filesystem::is_directory(odp) || !std::filesystem::exists(odp)) std::filesystem::create_directory(odp);
    std::filesystem::path odp2(outputDirectory+"/"+directory_snapshots);
    if(!std::filesystem::is_directory(odp2) || !std::filesystem::exists(odp2)) std::filesystem::create_directory(odp2);

    const int current_frame_number = model->prms.AnimationFrameNumber();
    char fileName[20];
    snprintf(fileName, sizeof(fileName), "d%05d.h5", current_frame_number);
    std::string filePath = outputDirectory + "/" + directory_snapshots + "/" + fileName;
    spdlog::info("saving NC frame {} to file {}", current_frame_number, filePath);

    H5::H5File file(filePath, H5F_ACC_TRUNC);

    hsize_t dims_indenter = model->prms.n_indenter_subdivisions*2;
    H5::DataSpace dataspace_indneter(1, &dims_indenter);
    H5::DataSet dataset_indneter = file.createDataSet("Indenter_2D", H5::PredType::NATIVE_DOUBLE, dataspace_indneter);
    dataset_indneter.write(model->gpu.host_side_indenter_force_accumulator, H5::PredType::NATIVE_DOUBLE);

    hsize_t dims_params = sizeof(icy::SimParams);
    H5::DataSpace dataspace_params(1, &dims_params);
    H5::DataSet dataset_params = file.createDataSet("Params", H5::PredType::NATIVE_B8, dataspace_params);
    dataset_params.write(&model->prms, H5::PredType::NATIVE_B8);

    SaveParametersAsAttributes(dataset_params);

    hsize_t dims_points = model->prms.nPtsPitch*icy::SimParams::nPtsArrays;
    H5::DataSpace dataspace_points(1, &dims_points);

//    hsize_t chunk_dims = (hsize_t)std::min(1024*256, model->prms.nPts);
    H5::DSetCreatPropList proplist;
//    proplist.setChunk(1, &chunk_dims);
//    proplist.setDeflate(4);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_DOUBLE, dataspace_points, proplist);
    dataset_points.write(model->gpu.tmp_transfer_buffer, H5::PredType::NATIVE_DOUBLE);

    file.close();



//    hsize_t att_dim = 1;
//    H5::DataSpace att_dspace(1, &att_dim);
//    H5::Attribute att = dataset_points.createAttribute("full_data", H5::PredType::NATIVE_INT,att_dspace);
//    att.write(H5::PredType::NATIVE_INT, &full_data);

*/
}

void icy::SnapshotManager::SaveParametersAsAttributes(H5::DataSet &dataset)
{
    /*
    hsize_t att_dim = 1;
    H5::DataSpace att_dspace(1, &att_dim);
    H5::Attribute att_indenter_x = dataset.createAttribute("indenter_x", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_indenter_y = dataset.createAttribute("indenter_y", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_SimulationTime = dataset.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_GridX = dataset.createAttribute("GridX", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_GridY = dataset.createAttribute("GridY", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_nPts = dataset.createAttribute("nPts", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_nPtsPitch = dataset.createAttribute("nPtsPitch", H5::PredType::NATIVE_INT64, att_dspace);

    H5::Attribute att_UpdateEveryNthStep = dataset.createAttribute("UpdateEveryNthStep", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_n_indenter_subdivisions = dataset.createAttribute("n_indenter_subdivisions", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_cellsize = dataset.createAttribute("cellsize", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_IndDiameter = dataset.createAttribute("IndDiameter", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_InitialTimeStep = dataset.createAttribute("InitialTimeStep", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_SetupType = dataset.createAttribute("SetupType", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_Volume = dataset.createAttribute("Volume", H5::PredType::NATIVE_DOUBLE, att_dspace);

    att_indenter_x.write(H5::PredType::NATIVE_DOUBLE, &model->prms.indenter_x);
    att_indenter_y.write(H5::PredType::NATIVE_DOUBLE, &model->prms.indenter_y);
    att_SimulationTime.write(H5::PredType::NATIVE_DOUBLE, &model->prms.SimulationTime);
    att_GridX.write(H5::PredType::NATIVE_INT, &model->prms.GridX);
    att_GridY.write(H5::PredType::NATIVE_INT, &model->prms.GridY);
    att_nPts.write(H5::PredType::NATIVE_INT, &model->prms.nPts);
    att_nPtsPitch.write(H5::PredType::NATIVE_INT64, &model->prms.nPtsPitch);

    att_UpdateEveryNthStep.write(H5::PredType::NATIVE_INT, &model->prms.UpdateEveryNthStep);
    att_n_indenter_subdivisions.write(H5::PredType::NATIVE_INT, &model->prms.n_indenter_subdivisions);
    att_cellsize.write(H5::PredType::NATIVE_DOUBLE, &model->prms.cellsize);
    att_IndDiameter.write(H5::PredType::NATIVE_DOUBLE, &model->prms.IndDiameter);
    att_InitialTimeStep.write(H5::PredType::NATIVE_DOUBLE, &model->prms.InitialTimeStep);
    att_SetupType.write(H5::PredType::NATIVE_INT, &model->prms.SetupType);
    att_Volume.write(H5::PredType::NATIVE_DOUBLE, &model->prms.Volume);
*/
}




void icy::SnapshotManager::ReadSnapshot(std::string fileName)
{
    /*
    if(!std::filesystem::exists(fileName)) return -1;

    std::string numbers = fileName.substr(fileName.length()-8,5);
    int idx = std::stoi(numbers);
    spdlog::info("reading snapshot {}", idx);

    H5::H5File file(fileName, H5F_ACC_RDONLY);

    // read and process SimParams
    H5::DataSet dataset_params = file.openDataSet("Params");
    hsize_t dims_params = 0;
    dataset_params.getSpace().getSimpleExtentDims(&dims_params, NULL);
    if(dims_params != sizeof(icy::SimParams)) throw std::runtime_error("ReadSnapshot: SimParams size mismatch");

    icy::SimParams tmp_params;
    dataset_params.read(&tmp_params, H5::PredType::NATIVE_B8);

    if(tmp_params.nGridPitch != model->prms.nGridPitch || tmp_params.nPtsPitch != model->prms.nPtsPitch)
        model->gpu.cuda_allocate_arrays(tmp_params.nGridPitch,tmp_params.nPtsPitch);
    double ParticleViewSize = model->prms.ParticleViewSize;
    model->prms = tmp_params;
    model->prms.ParticleViewSize = ParticleViewSize;

    // read point data
    H5::DataSet dataset_points = file.openDataSet("Points");
//    H5::Attribute att = dataset_points.openAttribute("full_data");
//    int full_data;
//    att.read(H5::PredType::NATIVE_INT, &full_data);

    dataset_points.read(model->gpu.tmp_transfer_buffer,H5::PredType::NATIVE_DOUBLE);

    model->gpu.transfer_ponts_to_host_finalize(model->points);
    file.close();
    return idx;*/
}


