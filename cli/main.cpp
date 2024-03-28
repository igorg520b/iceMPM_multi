#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <atomic>
#include <thread>
#include <chrono>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include "model.h"
#include "snapshotmanager.h"
#include "gpu_partition.h"



int main(int argc, char** argv)
{
    icy::Model model;
    icy::SnapshotManager snapshot;
    std::string snapshot_directory = "cm_snapshots";
    std::thread snapshot_thread;
    std::atomic<bool> request_terminate = false;
    std::atomic<bool> request_full_snapshot = false;


    // initialize the model

    model.gpu.initialize();
    model.Reset();
    snapshot.model = &model;

    // parse options
    cxxopts::Options options("Ice MPM", "CLI version of MPM simulation");

    options.add_options()
        ("file", "Configuration file", cxxopts::value<std::string>())
        ;
    options.parse_positional({"file"});

    auto option_parse_result = options.parse(argc, argv);

    if(option_parse_result.count("file"))
    {
        std::string params_file = option_parse_result["file"].as<std::string>();
        std::string pointCloudFile = model.prms.ParseFile(params_file);
        snapshot.LoadRawPoints(pointCloudFile);
    }

    // what to do once the data is available
    model.gpu.transfer_completion_callback = [&](){
        if(snapshot_thread.joinable()) snapshot_thread.join();
        snapshot_thread = std::thread([&](){
            int snapshot_number = model.prms.AnimationFrameNumber();
            if(request_terminate) { model.UnlockCycleMutex(); std::cout << "snapshot aborted\n"; return; }
            spdlog::info("cycle callback {}; compute_time_per_cycle {}", snapshot_number, model.compute_time_per_cycle);
            snapshot.SaveSnapshot(snapshot_directory);
            model.UnlockCycleMutex();
            spdlog::info("callback {} done", snapshot_number);
        });
    };

    // ensure that the folder exists
    std::filesystem::path outputFolder(snapshot_directory);
    std::filesystem::create_directory(outputFolder);

    /*
    // start the simulation thread
    std::thread simulation_thread([&](){
        bool result;
        do
        {
            result = model.Step();
        } while(!request_terminate && result);
    });

    // accept console input
    do
    {
        std::string user_input;
        std::cin >> user_input;

        if(user_input[0]=='s')
        {
            request_full_snapshot = true;
            spdlog::critical("requested to save a full snapshot");
        }
        else if(user_input[0]=='q'){
            request_terminate = true;
            request_full_snapshot = true;
            spdlog::critical("requested to save the snapshot and terminate");
        }
    } while(!request_terminate);

    simulation_thread.join();
    model.gpu.synchronize();
    snapshot_thread.join();
*/
    std::cout << "cm done\n";

    return 0;
}
