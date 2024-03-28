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

#include "parameters_sim.h"
#include "point.h"
#include "gpu_implementation5.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>


namespace icy { class Model; }

class icy::Model
{
public:
    Model();
    void Reset();

    void Prepare();        // invoked once, at simulation start
    bool Step();           // either invoked by Worker or via GUI
    void RequestAbort() {abortRequested = true;}   // asynchronous stop

    void UnlockCycleMutex();

    icy::SimParams prms;
    GPU_Implementation5 gpu;
    float compute_time_per_cycle;

    std::mutex processing_current_cycle_data; // locked until the current cycle results' are copied to host and processed

private:
    void ResetGrid();
    void P2G();
    void UpdateNodes();
    void G2P();
    void UpdateIndenterPosition(double simulationtime);

    bool abortRequested;
};

#endif
