//
// Created by Max Würfel on 25.06.23.
//

#include "SpikeTrain.h"
#include "STTC.h"
#include <string>
#include <H5Cpp.h>
#include <vector>
#include <iostream>

SpikeTrain::SpikeTrain(std::string FILE_NAME_,
                       std::vector<std::vector<double>> &electrodePos,
                       int numElectrodes,
                       double sttcCutoff
) {
    /**
     * This is the constructor for the spike train reading from HD5 specified in file name
     */
    FILE_NAME = FILE_NAME_;

    spikes = readDoubleDataset(FILE_NAME, "spikes");
    spikeCounts = readDoubleDataset(FILE_NAME, "sCount");
    numActiveElectrodes = spikeCounts.size();
    recordingTime = readDoubleDataset(FILE_NAME, "recordingtime");
    div = (int) readDoubleDataset(FILE_NAME, "meta/age")[0];
    // region = readByteStringDataset(FILE_NAME, "meta/region")[0];

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
    for (int i = 0; i < numActiveElectrodes; ++i) {
        auto it = std::find(electrodePos.begin(), electrodePos.end(), activeElectrodePos[i]);
        if (it != electrodePos.end()) {
            // TODO: Why would we ever hit the condition
            activeElectrodes.push_back(std::distance(electrodePos.begin(), it));
        }
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

std::vector<std::string> SpikeTrain::readByteStringDataset(std::string file_name,
                                                           std::string dataset_name) {
    /**
     * This member function reads a byte string dataset from the HDF5 file.
     */
    // Open the HDF5 file & dataset
    H5::H5File file(file_name, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(dataset_name);

    // Get the dataspace of the dataset
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t numElements = dataspace.getSimpleExtentNpoints();

    hsize_t dims[3];
    dataspace.getSimpleExtentDims(dims, NULL);
    std::cout << dims[0] << dims[1] << dims[2] << std::endl;

    // Define the datatype for the byte string
    H5::StrType dataType(H5::PredType::C_S1, 3);

    // Read the data from the dataset
    std::vector<char *> data(numElements);
    dataset.read(data.data(), dataType);

    // Convert char* to std::string
    std::vector<std::string> byteStrings(numElements);
    for (hsize_t i = 0; i < numElements; ++i) {
        byteStrings[i] = std::string(data[i]);
        delete[] data[i];  // Release memory allocated by HDF5 library
    }

    dataset.close();
    file.close();

    return byteStrings;
}


//int main() {
//    const std::string FILE_NAME = "/Users/maxwuerfek/code/diss/data/g2c_data/C57_CTX_G2CEPHYS3_TC21_DIV28_D.h5";
//    SpikeTrain test_st(FILE_NAME);
//    std::cout << 1 << std::endl;
//    return 0;
//}