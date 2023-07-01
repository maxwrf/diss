cmake_minimum_required(VERSION 3.25)
project(gnm-run)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")


add_executable(mainTest mainTest.cpp SpikeTrain.cpp SpikeTrain.h STTC.cpp STTC.h GNM.cpp GNM.h TestData.cpp TestData.h WeightedGNM.cpp)


target_include_directories(mainTest PRIVATE /opt/homebrew/include PRIVATE /opt/homebrew/include/eigen3/)


target_link_directories(mainTest PRIVATE /opt/homebrew/lib)


target_link_libraries(mainTest PRIVATE hdf5 hdf5_hl hdf5_cpp)