#include <iostream>
#include <string>
#include <H5Cpp.h>
#include <vector>

struct SpikeTrain
{
    std::string FILE_NAME;
    std::vector<double> spikes;
    std::vector<double> recordingTime;
    std::vector<double> epos;
    std::vector<std::string> electrodes;

    std::vector<double> readDoubleDataset(std::string file_name,
                                          std::string dataset_name);

    SpikeTrain(std::string FILE_NAME)
    {
        FILE_NAME = FILE_NAME;
        spikes = readDoubleDataset(FILE_NAME, "spikes");
        recordingTime = readDoubleDataset(FILE_NAME, "recordingtime");
        epos = readDoubleDataset(FILE_NAME, "epos");
    }
};

std::vector<double> SpikeTrain::readDoubleDataset(std::string file_name,
                                                  std::string dataset_name)
{
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

int main()
{
    const std::string FILE_NAME = "/Users/maxwuerfek/code/diss/data/g2c_data/C57_CTX_G2CEPHYS3_TC21_DIV28_D.h5";
    SpikeTrain test_st(FILE_NAME);

    return 0;
}
