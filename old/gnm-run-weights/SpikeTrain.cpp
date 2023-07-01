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
#include <iostream>

SpikeTrain::SpikeTrain(std::string FILE_NAME_,
                       std::vector<std::vector<double>> &electrodePos,
                       std::vector<int> &electrodes,
                       int numElectrodes,
                       double dt,
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
    recordingTime = {*std::min_element(spikes.begin(), spikes.end()),
                     *std::max_element(spikes.begin(), spikes.end())
    };

    // Spike sorting
    sttc = STTC::tiling(dt,
                        recordingTime,
                        spikes,
                        spikeCounts
    );

    // Get the active electrodes names and indices
    activeElectrodeNames = readByteStringDataset(FILE_NAME, "names");
    getActiveElectrodeNumbers(electrodes);

    // Constructing the matrices
    initAdjacencyMatrices(numElectrodes);
}

void SpikeTrain::initAdjacencyMatrices(int numElectrodes) {
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

std::vector<std::string> SpikeTrain::readByteStringDataset(std::string file_name,
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
    char *out = new char[n * data_size * dims_out[0]]();
    dataset.read(out, data_type);

    // Read in the string
    std::vector<std::string> data(dims_out[0]);
    auto len = data_size;
    for (auto i = 0u; i < dims_out[0]; ++i) {
        auto c_str = out + data_size * i;
        for (auto p = c_str + len - 1; p != c_str && !*p; --p) --len;
        data[i] = c_str;
    }
    return data;
}

void SpikeTrain::getGroupId(int dSet) {
    /**
     * The group ID needs to identify a subset of the HD5 spike trains in the provided directory
     * Will require specification for different spike train sets
     */
    switch (dSet) {
        int div;
        case 0:
        {
            div = (int) readDoubleDataset(FILE_NAME, "meta/age")[0];
            std::string region = readByteStringDataset(FILE_NAME, "meta/region")[0];
            groupId = region + std::to_string(div);
            break;
        }

        case 1:
        {
            div = (int) readDoubleDataset(FILE_NAME, "meta/age")[0];
            groupId = std::to_string(div);
            break;
        }

        case 2:
        {
            div = (int) readDoubleDataset(FILE_NAME, "meta/age")[0];
            groupId = std::to_string(div);
            break;
        }

        default:
            std::cout << "No group ID implemented for this dataset" << std::endl;
    };
}

void SpikeTrain::getActiveElectrodeNumbers(std::vector<int> &electrodes) {
    // Extract the electrode position indicator
    std::vector<int> activeElectrodeNumbers;
    std::vector<int> removalElectrodes;
    for (int i = 0; i < numActiveElectrodes; ++i) {
        if (activeElectrodeNames[i][5] == 'a') {
            std::string numberString = activeElectrodeNames[i].substr(3, 2);
            activeElectrodeNumbers.push_back(std::stoi(numberString));
        } else {
            // Handle the case when the same electrode was allocated multiple spikes
            removalElectrodes.push_back(i);
        };
    }

    // Clean up the removed spikes at some electrodes
    numActiveElectrodes -= removalElectrodes.size();
    for (int i = 0; i < removalElectrodes.size(); ++i) {
        activeElectrodeNames.erase(activeElectrodeNames.begin() + removalElectrodes[i] - i);
        sttc.erase(sttc.begin() + removalElectrodes[i] - i);
    }

    // Find the index of that positon
    for (int i = 0; i < numActiveElectrodes; ++i) {
        auto it = std::find(electrodes.begin(), electrodes.end(), activeElectrodeNumbers[i]);
        if (it != electrodes.end()) {
            activeElectrodes.push_back(std::distance(electrodes.begin(), it));
        }
    }
}
