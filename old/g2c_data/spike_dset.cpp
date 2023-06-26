#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <H5Cpp.h>

using namespace std;

class SpikeData
{
private:
    string path;
    vector<string> hd5Files;

    static void attr_op(H5::H5Location &loc, const std::string attr_name,
                        void *operator_data)
    {
        std::cout << attr_name << std::endl;
    }

public:
    SpikeData(string &path_);
    ~SpikeData();
};

SpikeData::SpikeData(string &path_)
{
    path = path_;

    // read in the files
    for (const auto &entry : filesystem::directory_iterator(path))
    {
        if (filesystem::is_regular_file(entry.path()))
        {
            string f = entry.path().filename().string();

            if (f.substr(f.find_last_of(".") + 1) == "h5")
            {
                hd5Files.push_back(f);

                // read in the meta data
                string file_name = "/Users/maxwuerfek/code/diss/data/g2c_data/C57_CTX_G2CEPHYS3_TC11_DIV21_C.h5";
                string dataset_name = "meta";
                H5::H5File file(file_name, H5F_ACC_RDONLY);
                auto dataset = file.openDataSet(dataset_name);
                dataset.iterateAttrs((H5::attr_operator_t)attr_op);
                int xy = 1;
            }
        }
    }
}

SpikeData::~SpikeData()
{
}

int main()
{
    string p = "/Users/maxwuerfek/code/diss/data/g2c_data";
    SpikeData spike_data(p);
}

// code to read h5 file