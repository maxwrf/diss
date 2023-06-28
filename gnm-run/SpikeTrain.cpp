//
// Created by Max Würfel on 25.06.23.
//

#include "SpikeTrain.h"
#include "STTC.h"
#include <string>
#include <H5Cpp.h>
#include <vector>
#include <cassert>
#include <algorithm>

SpikeTrain::SpikeTrain(std::string FILE_NAME_,
                       std::vector<std::vector<double>> &electrodePos,
                       int numElectrodes,
                       double sttcCutoff,
                       int dSet
) {
    /**
     * This is the constructor for the spike train reading from HD5 specified in file name
     */
    FILE_NAME = FILE_NAME_;
    getGroupId(dSet);

    spikes = readDoubleDataset(FILE_NAME, "spikes");
    spikeCounts = readDoubleDataset(FILE_NAME, "sCount");
    numActiveElectrodes = spikeCounts.size();
    recordingTime = {0, readDoubleDataset(FILE_NAME, "/summary/duration")[0]};


    // Get the electrode Positions
    std::vector<double> temp = readDoubleDataset(FILE_NAME, "epos");
    activeElectrodePos = std::vector<std::vector<double>>(
            numActiveElectrodes,
            std::vector<double>(2)
    );
    for (int i = 0; i < numActiveElectrodes; ++i) {
        for (int j = 0; j < 2; ++j) {
            activeElectrodePos[i][j] = temp[i + numActiveElectrodes * j];
        }
    }

    // Find the active electrodes
    bool fix = false;
    for (int i = 0; i < numActiveElectrodes; ++i) {
        auto it = std::find(electrodePos.begin(), electrodePos.end(), activeElectrodePos[i]);
        if (it != electrodePos.end()) {
            activeElectrodes.push_back(std::distance(electrodePos.begin(), it));
        } else {
            // TODO: Why would we ever hit the condition, this must be an error
            fix = true;
        }
    }

    // TODO: This is only temp fix
    if (fix) {
        numActiveElectrodes--;
    }

    // Spike sorting
    sttc = STTC::tiling(0.05,
                        recordingTime,
                        spikes,
                        spikeCounts
    );

    // Construct A with one where sttc threshold is met, and A init as subset
    A_Y = std::vector<std::vector<double>>(
            numElectrodes,
            std::vector<double>(numElectrodes, 0.0)
    );

    A_init = std::vector<std::vector<double>>(
            numElectrodes,
            std::vector<double>(numElectrodes, 0.0)
    );

    m = 0;
    for (int i = 0; i < numActiveElectrodes; ++i) {
        for (int j = i + 1; j < numActiveElectrodes; ++j) {
            if (sttc[i][j] > sttcCutoff) {
                int iElectrode = activeElectrodes[i];
                int jElectrode = activeElectrodes[j];
                A_Y[iElectrode][jElectrode] = A_Y[jElectrode][iElectrode] = 1;
                m++;

                // for initialization
                // TODO: This does not ensure always exactly 20 percent
                if ((static_cast<double>(std::rand()) / RAND_MAX) < 0.2) {
                    A_init[iElectrode][jElectrode] = A_init[jElectrode][iElectrode] = 1;
                }
            }
        }
    }
}

std::vector<double> SpikeTrain::readDoubleDataset(std::string file_name,
                                                  std::string dataset_name) {
    /**
     * This member function reads a double dataset from the HD5
     */
    // Open the HDF5 file & dataset
    H5::H5File file(file_name, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(dataset_name);

    // Get the dataspace of the dataset
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t numElements = dataspace.getSimpleExtentNpoints();

    // Read the data from the dataset
    std::vector<double> data(numElements);
    dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);

    dataset.close();
    file.close();

    return data;
}

std::string SpikeTrain::readByteString(std::string file_name,
                                       std::string dataset_name) {
    /**
     * This member function reads a byte string dataset from the HDF5 file.
     */

    // Open the dataspace
    H5::H5File file(file_name, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(dataset_name);

    // Check for the dimensions
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims_out[2];
    auto n = dataspace.getSimpleExtentDims(dims_out, nullptr);
    assert(n == 1);

    // Get the datatype and the size of the string
    auto data_type = dataset.getDataType();
    auto type_class = data_type.getClass();
    auto data_size = data_type.getSize();

    // prepare a buffer and read in
    char *out = new char[n * data_size]();
    dataset.read(out, data_type);

    // Convert to std::string
    std::string *strs = new std::string[n];
    for (auto i = 0u; i < n; ++i) {
        auto len = data_size;
        auto c_str = out + data_size * i;
        for (auto p = c_str + len - 1; p != c_str && !*p; --p) --len;
        strs[i].assign(c_str, len);
    }

    return *strs;
}

void SpikeTrain::getGroupId(int dSet) {
    /**
     * The group ID needs to identify a subset of the HD5 spike trains in the provided directory
     * Will require specification for different spike train sets
     */
    if (dSet == 0) {
        int div = (int) readDoubleDataset(FILE_NAME, "meta/age")[0];
        std::string region = readByteString(FILE_NAME, "meta/region");
        groupId = region + std::to_string(div);
    } else if (dSet == 1) {
        int div = (int) readDoubleDataset(FILE_NAME, "meta/age")[0];
        groupId = std::to_string(div);
    }

}


//int main() {
//    const std::string FILE_NAME = "/Users/maxwuerfek/code/diss/data/g2c_data/C57_CTX_G2CEPHYS3_TC21_DIV28_D.h5";
//    SpikeTrain test_st(FILE_NAME);
//    std::cout << 1 << std::endl;
//    return 0;
//}