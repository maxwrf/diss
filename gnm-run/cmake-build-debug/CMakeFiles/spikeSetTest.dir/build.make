# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /mhome/damtp/r/mw894/.cache/JetBrains/RemoteDev/dist/036938132e60c_CLion-232.8296.18/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /mhome/damtp/r/mw894/.cache/JetBrains/RemoteDev/dist/036938132e60c_CLion-232.8296.18/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mhome/damtp/r/mw894/diss/gnm-run

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/spikeSetTest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/spikeSetTest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/spikeSetTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/spikeSetTest.dir/flags.make

CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.o: CMakeFiles/spikeSetTest.dir/flags.make
CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.o: /mhome/damtp/r/mw894/diss/gnm-run/SpikeSet.cpp
CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.o: CMakeFiles/spikeSetTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.o"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.o -MF CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.o.d -o CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.o -c /mhome/damtp/r/mw894/diss/gnm-run/SpikeSet.cpp

CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mhome/damtp/r/mw894/diss/gnm-run/SpikeSet.cpp > CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.i

CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mhome/damtp/r/mw894/diss/gnm-run/SpikeSet.cpp -o CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.s

CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.o: CMakeFiles/spikeSetTest.dir/flags.make
CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.o: /mhome/damtp/r/mw894/diss/gnm-run/SpikeTrain.cpp
CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.o: CMakeFiles/spikeSetTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.o"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.o -MF CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.o.d -o CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.o -c /mhome/damtp/r/mw894/diss/gnm-run/SpikeTrain.cpp

CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mhome/damtp/r/mw894/diss/gnm-run/SpikeTrain.cpp > CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.i

CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mhome/damtp/r/mw894/diss/gnm-run/SpikeTrain.cpp -o CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.s

CMakeFiles/spikeSetTest.dir/STTC.cpp.o: CMakeFiles/spikeSetTest.dir/flags.make
CMakeFiles/spikeSetTest.dir/STTC.cpp.o: /mhome/damtp/r/mw894/diss/gnm-run/STTC.cpp
CMakeFiles/spikeSetTest.dir/STTC.cpp.o: CMakeFiles/spikeSetTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/spikeSetTest.dir/STTC.cpp.o"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/spikeSetTest.dir/STTC.cpp.o -MF CMakeFiles/spikeSetTest.dir/STTC.cpp.o.d -o CMakeFiles/spikeSetTest.dir/STTC.cpp.o -c /mhome/damtp/r/mw894/diss/gnm-run/STTC.cpp

CMakeFiles/spikeSetTest.dir/STTC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spikeSetTest.dir/STTC.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mhome/damtp/r/mw894/diss/gnm-run/STTC.cpp > CMakeFiles/spikeSetTest.dir/STTC.cpp.i

CMakeFiles/spikeSetTest.dir/STTC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spikeSetTest.dir/STTC.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mhome/damtp/r/mw894/diss/gnm-run/STTC.cpp -o CMakeFiles/spikeSetTest.dir/STTC.cpp.s

# Object files for target spikeSetTest
spikeSetTest_OBJECTS = \
"CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.o" \
"CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.o" \
"CMakeFiles/spikeSetTest.dir/STTC.cpp.o"

# External object files for target spikeSetTest
spikeSetTest_EXTERNAL_OBJECTS =

spikeSetTest: CMakeFiles/spikeSetTest.dir/SpikeSet.cpp.o
spikeSetTest: CMakeFiles/spikeSetTest.dir/SpikeTrain.cpp.o
spikeSetTest: CMakeFiles/spikeSetTest.dir/STTC.cpp.o
spikeSetTest: CMakeFiles/spikeSetTest.dir/build.make
spikeSetTest: CMakeFiles/spikeSetTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable spikeSetTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spikeSetTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/spikeSetTest.dir/build: spikeSetTest
.PHONY : CMakeFiles/spikeSetTest.dir/build

CMakeFiles/spikeSetTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/spikeSetTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/spikeSetTest.dir/clean

CMakeFiles/spikeSetTest.dir/depend:
	cd /mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mhome/damtp/r/mw894/diss/gnm-run /mhome/damtp/r/mw894/diss/gnm-run /mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug /mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug /mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles/spikeSetTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/spikeSetTest.dir/depend

