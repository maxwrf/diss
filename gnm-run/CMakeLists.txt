cmake_minimum_required(VERSION 3.25)
project(gnm-run)

set(CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp SpikeSet.cpp SpikeSet.h SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h GNM.cpp GNM.h)
add_executable(spikeTrainTest SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h)
add_executable(spikeSetTest SpikeSet.cpp SpikeSet.h SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h)
add_executable(sttcTest STTC.cpp STTC.h)
add_executable(syntheticTest syntheticTest.cpp SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h GNM.cpp GNM.h)

target_include_directories(spikeTrainTest PRIVATE /opt/homebrew/include)
target_include_directories(spikeSetTest PRIVATE /opt/homebrew/include)
target_include_directories(main PRIVATE /opt/homebrew/include)
target_include_directories(syntheticTest PRIVATE /opt/homebrew/include)

target_link_directories(spikeTrainTest PRIVATE /opt/homebrew/lib)
target_link_directories(spikeSetTest PRIVATE /opt/homebrew/lib)
target_link_directories(main PRIVATE /opt/homebrew/lib)
target_link_directories(syntheticTest PRIVATE /opt/homebrew/lib)

target_link_libraries(spikeTrainTest PRIVATE hdf5 hdf5_hl hdf5_cpp)
target_link_libraries(spikeSetTest PRIVATE hdf5 hdf5_hl hdf5_cpp)
target_link_libraries(main PRIVATE hdf5 hdf5_hl hdf5_cpp)
target_link_libraries(syntheticTest PRIVATE hdf5 hdf5_hl hdf5_cpp)
