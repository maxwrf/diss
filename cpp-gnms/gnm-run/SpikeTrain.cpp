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
                       std::vector<std::vector<int>> &electrodes,
                       int numElectrodes,
                       double dt,
                       double sttcCutoff_,
                       int dSet,
                       int meaType_
) {
    /**
     * This is the constructor for the spike train reading from HD5 specified in file name
     */

    // Initizalize the class variables
    FILE_NAME = FILE_NAME_;
    sttcCutoff = sttcCutoff_;
    meaType = meaType_;

    // Read the data from HD5
    getGroupId(dSet);
    spikes = readDoubleDataset(FILE_NAME, "spikes");
    spikeCounts = readDoubleDataset(FILE_NAME, "sCount");
    numActiveElectrodes = spikeCounts.size();
    recordingTime = {*std::min_element(spikes.begin(), spikes.end()),
                     *std::max_element(spikes.begin(), spikes.end())
    };

    // Compute STTC
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
    int asdas = 0;
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
                int iElectrode = activeElectrodeIdx[i];
                int jElectrode = activeElectrodeIdx[j];
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
        case 0: {
            div = (int) readDoubleDataset(FILE_NAME, "meta/age")[0];
            std::string region = readByteStringDataset(FILE_NAME, "meta/region")[0];
            groupId = region + std::to_string(div);
            break;
        }

        case 1:
        case 2:
        case 3: {
            div = (int) readDoubleDataset(FILE_NAME, "meta/age")[0];
            groupId = std::to_string(div);
            break;
        }

        default:
            std::cout << "No group ID implemented for this dataset" << std::endl;
    };
}

void SpikeTrain::getActiveElectrodeNumbers(std::vector<std::vector<int>> &electrodes) {
    // Extract the electrode position indicator, and those to remove if not A
    std::vector<std::vector<int>> activeElectrodes;
    std::vector<int> removalElectrodesIdx;
    switch (meaType) {
        case 0:
        case 1: {
            // For MCS_8x8_200um and MCS_8x8_100um, the format of the spike names is e.g., ch_12B_unit_0
            for (int i = 0; i < numActiveElectrodes; ++i) {
                if (activeElectrodeNames[i][5] == 'a' | activeElectrodeNames[i][5] == 'A') {
                    int x = std::stoi(activeElectrodeNames[i].substr(3, 1));
                    int y = std::stoi(activeElectrodeNames[i].substr(4, 1));
                    activeElectrodes.push_back({x, y});
                } else {
                    // Handle the case when the same electrode was allocated multiple spikes
                    removalElectrodesIdx.push_back(i);
                };
            }
            break;
        }
        case 2: {
            // For the APS_64x64_42um, the format of the spike names is e.g., Ch11.14, there is no A, B
            for (int i = 0; i < numActiveElectrodes; ++i) {
                std::string numberString = activeElectrodeNames[i].substr(2);
                size_t dotPosition = numberString.find('.');
                int x = std::stoi(numberString.substr(0, dotPosition));
                int y = std::stoi(numberString.substr(dotPosition + 1));

                // TODO: sometimes there is an electrode pos at 65?
                if ((x > 64) | (y > 64)) {
                    removalElectrodesIdx.push_back(i);
                } else {
                    activeElectrodes.push_back({x, y});
                }
            }
            break;
        }
    }

    // Clean up the removed spikes at some electrodes
    if (removalElectrodesIdx.size() > 0) {
        numActiveElectrodes -= removalElectrodesIdx.size();
        for (int i = 0; i < removalElectrodesIdx.size(); ++i) {
            activeElectrodeNames.erase(activeElectrodeNames.begin() + removalElectrodesIdx[i] - i);
            sttc.erase(sttc.begin() + removalElectrodesIdx[i] - i);
        }
    }

    // Find the index of these electrodes in all the electrodes on this array
    for (int i = 0; i < numActiveElectrodes; ++i) {
        auto it = std::find(electrodes.begin(), electrodes.end(), activeElectrodes[i]);
        if (it != electrodes.end()) {
            activeElectrodeIdx.push_back(std::distance(electrodes.begin(), it));
        } else {
            throw std::runtime_error("Read in an electrode which is not in the MEA");
        }
    }
}
