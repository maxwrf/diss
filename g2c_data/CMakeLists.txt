cmake_minimum_required(VERSION 3.19)
project(diss)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(MyExecutable ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_include_directories(MyExecutable PRIVATE
        /opt/homebrew/include
        /Users/maxwuerfek/code/diss/diss-env/include/python3.8
        /Users/maxwuerfek/code/diss/diss-env/lib/python3.8/site-packages/numpy/core/include
        )

target_link_directories(MyExecutable PRIVATE
        /opt/homebrew/lib
        )

target_link_libraries(MyExecutable PRIVATE
        hdf5
        hdf5_hl
        hdf5_cpp
        )
