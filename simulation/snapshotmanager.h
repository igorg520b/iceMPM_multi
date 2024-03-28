#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <array>
#include <vector>
#include <string>

#include <H5Cpp.h>

namespace icy {class SnapshotManager; class Model;}


class icy::SnapshotManager
{
public:
    icy::Model *model;

    void SaveSnapshot(std::string directory);
    void ReadSnapshot(std::string fileName); // return file number
    void LoadRawPoints(std::string fileName);
    void SaveParametersAsAttributes(H5::DataSet &dataset);
    void SortPoints(std::vector<std::tuple<float,float,short>> &buffer);
    void SavePQ(std::string directory);

    const std::string directory_snapshots = "snapshots";
    const std::string directory_pq = "pq";
    std::vector<double> buffer1, buffer2;
};

#endif // SNAPSHOTWRITER_H
