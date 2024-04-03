#include "model.h"
#include <spdlog/spdlog.h>


icy::Model::Model()
{
    prms.Reset();
    gpu.model = this;
    GPU_Partition::prms = &this->prms;
    spdlog::info("Model constructor");
};

bool icy::Model::Step()
{
    double simulation_time = prms.SimulationTime;
    std::cout << '\n';
    spdlog::info("step {} ({}) started; sim_time {:.3}", prms.SimulationStep, prms.SimulationStep/prms.UpdateEveryNthStep, simulation_time);
    int count_unupdated_steps = 0;

    gpu.reset_indenter_force_accumulator();

    gpu.reset_grid();
    gpu.p2g();

    /*

    if(prms.SimulationStep % (prms.UpdateEveryNthStep*2) == 0) cudaEventRecord(gpu.eventCycleStart);
    do
    {
        count_unupdated_steps++;
        simulation_time += prms.InitialTimeStep;
        prms.indenter_x = prms.indenter_x_initial + simulation_time*prms.IndVelocity;
        gpu.cuda_reset_grid();
        gpu.cuda_p2g();
        gpu.cuda_update_nodes(prms.indenter_x, prms.indenter_y);
        gpu.cuda_g2p((prms.SimulationStep+count_unupdated_steps) % prms.UpdateEveryNthStep == 0);
    } while((prms.SimulationStep+count_unupdated_steps) % prms.UpdateEveryNthStep != 0);
    if(prms.SimulationStep % (prms.UpdateEveryNthStep*2) == 0) cudaEventRecord(gpu.eventCycleStop);

    processing_current_cycle_data.lock();   // if locked, previous results are not yet processed by the host

    gpu.cuda_transfer_from_device();

    if(prms.SimulationStep % (prms.UpdateEveryNthStep*2) != 0)
    {
        cudaEventSynchronize(gpu.eventCycleStop);
        cudaEventElapsedTime(&compute_time_per_cycle, gpu.eventCycleStart, gpu.eventCycleStop);
        compute_time_per_cycle /= prms.UpdateEveryNthStep;
        spdlog::info("cycle time {:.3} ms", compute_time_per_cycle);
    }

    prms.SimulationTime = simulation_time;
    prms.SimulationStep += count_unupdated_steps;
    return (prms.SimulationTime < prms.SimulationEndTime && !gpu.error_code);
*/
    return true;
}


void icy::Model::UnlockCycleMutex()
{
    // current data was handled by host - allow next cycle to proceed
    processing_current_cycle_data.unlock();
}


void icy::Model::Reset()
{
    spdlog::info("icy::Model::Reset()");

    prms.SimulationStep = 0;
    prms.SimulationTime = 0;
    compute_time_per_cycle = 0;
}


void icy::Model::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    abortRequested = false;
    gpu.update_constants();
}

