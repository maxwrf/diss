cmake_minimum_required(VERSION 3.25)
project(gnm-run)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

# Test for weight model only locally
add_executable(mainTestWeights mainTestWeights.cpp SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h GNM.cpp GNM.h TestDataWeights.cpp TestDataWeights.h WeightedGNM.cpp)
target_include_directories(mainTestWeights PRIVATE /opt/homebrew/include PRIVATE /opt/homebrew/include/eigen3/)
target_link_directories(mainTestWeights PRIVATE /opt/homebrew/lib)
target_link_libraries(mainTestWeights PRIVATE hdf5 hdf5_hl hdf5_cpp)

# Test for std model only locally
add_executable(mainTest mainTest.cpp SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h GNM.cpp GNM.h TestData.cpp TestData.h)
target_include_directories(mainTest PRIVATE /opt/homebrew/include)
target_link_directories(mainTest PRIVATE /opt/homebrew/lib)
target_link_libraries(mainTest PRIVATE hdf5 hdf5_hl hdf5_cpp)

add_executable(spikeTrainTest SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h)
add_executable(spikeSetTest SpikeSet.cpp SpikeSet.h SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h)
add_executable(sttcTest STTC.cpp STTC.h)
add_executable(slurmPrep mainSlurmPrep.cpp Slurm.cpp Slurm.h SpikeSet.cpp SpikeSet.h SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h GNM.cpp GNM.h)
add_executable(slurmRun mainSlurmRun.cpp Slurm.cpp Slurm.h GNM.cpp GNM.h SpikeSet.cpp SpikeSet.h SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h)


target_include_directories(spikeTrainTest PRIVATE /opt/homebrew/include)
target_include_directories(spikeSetTest PRIVATE /opt/homebrew/include)
target_include_directories(slurmPrep PRIVATE /opt/homebrew/include)
target_include_directories(slurmPrep PRIVATE /home/mw894/hdf/CMake-hdf5-1.8.23/build/HDF5-1.8.23-Linux/HDF_Group/HDF5/1.8.23/include)
target_include_directories(slurmRun PRIVATE /opt/homebrew/include)
target_include_directories(slurmRun PRIVATE /home/mw894/hdf/CMake-hdf5-1.8.23/build/HDF5-1.8.23-Linux/HDF_Group/HDF5/1.8.23/include)

target_link_directories(spikeTrainTest PRIVATE /opt/homebrew/lib)
target_link_directories(spikeSetTest PRIVATE /opt/homebrew/lib)
target_link_directories(slurmPrep PRIVATE /opt/homebrew/lib)
target_link_directories(slurmPrep PRIVATE /home/mw894/hdf/CMake-hdf5-1.8.23/build/HDF5-1.8.23-Linux/HDF_Group/HDF5/1.8.23/lib)
target_link_directories(slurmRun PRIVATE /opt/homebrew/lib)
target_link_directories(slurmRun PRIVATE /home/mw894/hdf/CMake-hdf5-1.8.23/build/HDF5-1.8.23-Linux/HDF_Group/HDF5/1.8.23/lib)

target_link_libraries(spikeTrainTest PRIVATE hdf5 hdf5_hl hdf5_cpp)
target_link_libraries(spikeSetTest PRIVATE hdf5 hdf5_hl hdf5_cpp)
target_link_libraries(slurmPrep PRIVATE hdf5 hdf5_hl hdf5_cpp)
target_link_libraries(slurmRun PRIVATE hdf5 hdf5_hl hdf5_cpp)
