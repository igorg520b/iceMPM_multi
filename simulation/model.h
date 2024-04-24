#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <random>
#include <mutex>
#include <iostream>
#include <string>
#include <fstream>

#include "parameters_sim.h"
#include "point.h"
#include "gpu_implementation5.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/logger.h>


namespace icy { class Model; }

class icy::Model
{
public:
    Model();
    ~Model();
    void Reset();

    void Prepare();        // invoked once, at simulation start
    bool Step();           // either invoked by Worker or via GUI
    void RequestAbort() {abortRequested = true;}   // asynchronous stop

    void UnlockCycleMutex();

    icy::SimParams prms;
    GPU_Implementation5 gpu;
    int max_points_transferred, max_pt_deviation;
    bool SyncTopologyRequired;

    std::mutex processing_current_cycle_data; // locked until the current cycle results' are copied to host and processed
    std::mutex accessing_point_data;

private:
    bool abortRequested;
    std::shared_ptr<spdlog::logger> log_timing;
    std::shared_ptr<spdlog::logger> log_indenter_force;
};

#endif
