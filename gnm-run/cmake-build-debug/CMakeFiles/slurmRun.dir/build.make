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
include CMakeFiles/slurmRun.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/slurmRun.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/slurmRun.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/slurmRun.dir/flags.make

CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.o: CMakeFiles/slurmRun.dir/flags.make
CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.o: /mhome/damtp/r/mw894/diss/gnm-run/mainSlurmRun.cpp
CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.o: CMakeFiles/slurmRun.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.o"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.o -MF CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.o.d -o CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.o -c /mhome/damtp/r/mw894/diss/gnm-run/mainSlurmRun.cpp

CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mhome/damtp/r/mw894/diss/gnm-run/mainSlurmRun.cpp > CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.i

CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mhome/damtp/r/mw894/diss/gnm-run/mainSlurmRun.cpp -o CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.s

CMakeFiles/slurmRun.dir/Slurm.cpp.o: CMakeFiles/slurmRun.dir/flags.make
CMakeFiles/slurmRun.dir/Slurm.cpp.o: /mhome/damtp/r/mw894/diss/gnm-run/Slurm.cpp
CMakeFiles/slurmRun.dir/Slurm.cpp.o: CMakeFiles/slurmRun.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/slurmRun.dir/Slurm.cpp.o"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/slurmRun.dir/Slurm.cpp.o -MF CMakeFiles/slurmRun.dir/Slurm.cpp.o.d -o CMakeFiles/slurmRun.dir/Slurm.cpp.o -c /mhome/damtp/r/mw894/diss/gnm-run/Slurm.cpp

CMakeFiles/slurmRun.dir/Slurm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slurmRun.dir/Slurm.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mhome/damtp/r/mw894/diss/gnm-run/Slurm.cpp > CMakeFiles/slurmRun.dir/Slurm.cpp.i

CMakeFiles/slurmRun.dir/Slurm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slurmRun.dir/Slurm.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mhome/damtp/r/mw894/diss/gnm-run/Slurm.cpp -o CMakeFiles/slurmRun.dir/Slurm.cpp.s

CMakeFiles/slurmRun.dir/GNM.cpp.o: CMakeFiles/slurmRun.dir/flags.make
CMakeFiles/slurmRun.dir/GNM.cpp.o: /mhome/damtp/r/mw894/diss/gnm-run/GNM.cpp
CMakeFiles/slurmRun.dir/GNM.cpp.o: CMakeFiles/slurmRun.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/slurmRun.dir/GNM.cpp.o"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/slurmRun.dir/GNM.cpp.o -MF CMakeFiles/slurmRun.dir/GNM.cpp.o.d -o CMakeFiles/slurmRun.dir/GNM.cpp.o -c /mhome/damtp/r/mw894/diss/gnm-run/GNM.cpp

CMakeFiles/slurmRun.dir/GNM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slurmRun.dir/GNM.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mhome/damtp/r/mw894/diss/gnm-run/GNM.cpp > CMakeFiles/slurmRun.dir/GNM.cpp.i

CMakeFiles/slurmRun.dir/GNM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slurmRun.dir/GNM.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mhome/damtp/r/mw894/diss/gnm-run/GNM.cpp -o CMakeFiles/slurmRun.dir/GNM.cpp.s

CMakeFiles/slurmRun.dir/SpikeSet.cpp.o: CMakeFiles/slurmRun.dir/flags.make
CMakeFiles/slurmRun.dir/SpikeSet.cpp.o: /mhome/damtp/r/mw894/diss/gnm-run/SpikeSet.cpp
CMakeFiles/slurmRun.dir/SpikeSet.cpp.o: CMakeFiles/slurmRun.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/slurmRun.dir/SpikeSet.cpp.o"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/slurmRun.dir/SpikeSet.cpp.o -MF CMakeFiles/slurmRun.dir/SpikeSet.cpp.o.d -o CMakeFiles/slurmRun.dir/SpikeSet.cpp.o -c /mhome/damtp/r/mw894/diss/gnm-run/SpikeSet.cpp

CMakeFiles/slurmRun.dir/SpikeSet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slurmRun.dir/SpikeSet.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mhome/damtp/r/mw894/diss/gnm-run/SpikeSet.cpp > CMakeFiles/slurmRun.dir/SpikeSet.cpp.i

CMakeFiles/slurmRun.dir/SpikeSet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slurmRun.dir/SpikeSet.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mhome/damtp/r/mw894/diss/gnm-run/SpikeSet.cpp -o CMakeFiles/slurmRun.dir/SpikeSet.cpp.s

CMakeFiles/slurmRun.dir/SpikeTrain.cpp.o: CMakeFiles/slurmRun.dir/flags.make
CMakeFiles/slurmRun.dir/SpikeTrain.cpp.o: /mhome/damtp/r/mw894/diss/gnm-run/SpikeTrain.cpp
CMakeFiles/slurmRun.dir/SpikeTrain.cpp.o: CMakeFiles/slurmRun.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/slurmRun.dir/SpikeTrain.cpp.o"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/slurmRun.dir/SpikeTrain.cpp.o -MF CMakeFiles/slurmRun.dir/SpikeTrain.cpp.o.d -o CMakeFiles/slurmRun.dir/SpikeTrain.cpp.o -c /mhome/damtp/r/mw894/diss/gnm-run/SpikeTrain.cpp

CMakeFiles/slurmRun.dir/SpikeTrain.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slurmRun.dir/SpikeTrain.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mhome/damtp/r/mw894/diss/gnm-run/SpikeTrain.cpp > CMakeFiles/slurmRun.dir/SpikeTrain.cpp.i

CMakeFiles/slurmRun.dir/SpikeTrain.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slurmRun.dir/SpikeTrain.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mhome/damtp/r/mw894/diss/gnm-run/SpikeTrain.cpp -o CMakeFiles/slurmRun.dir/SpikeTrain.cpp.s

CMakeFiles/slurmRun.dir/STTC.cpp.o: CMakeFiles/slurmRun.dir/flags.make
CMakeFiles/slurmRun.dir/STTC.cpp.o: /mhome/damtp/r/mw894/diss/gnm-run/STTC.cpp
CMakeFiles/slurmRun.dir/STTC.cpp.o: CMakeFiles/slurmRun.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/slurmRun.dir/STTC.cpp.o"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/slurmRun.dir/STTC.cpp.o -MF CMakeFiles/slurmRun.dir/STTC.cpp.o.d -o CMakeFiles/slurmRun.dir/STTC.cpp.o -c /mhome/damtp/r/mw894/diss/gnm-run/STTC.cpp

CMakeFiles/slurmRun.dir/STTC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slurmRun.dir/STTC.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mhome/damtp/r/mw894/diss/gnm-run/STTC.cpp > CMakeFiles/slurmRun.dir/STTC.cpp.i

CMakeFiles/slurmRun.dir/STTC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slurmRun.dir/STTC.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mhome/damtp/r/mw894/diss/gnm-run/STTC.cpp -o CMakeFiles/slurmRun.dir/STTC.cpp.s

# Object files for target slurmRun
slurmRun_OBJECTS = \
"CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.o" \
"CMakeFiles/slurmRun.dir/Slurm.cpp.o" \
"CMakeFiles/slurmRun.dir/GNM.cpp.o" \
"CMakeFiles/slurmRun.dir/SpikeSet.cpp.o" \
"CMakeFiles/slurmRun.dir/SpikeTrain.cpp.o" \
"CMakeFiles/slurmRun.dir/STTC.cpp.o"

# External object files for target slurmRun
slurmRun_EXTERNAL_OBJECTS =

slurmRun: CMakeFiles/slurmRun.dir/mainSlurmRun.cpp.o
slurmRun: CMakeFiles/slurmRun.dir/Slurm.cpp.o
slurmRun: CMakeFiles/slurmRun.dir/GNM.cpp.o
slurmRun: CMakeFiles/slurmRun.dir/SpikeSet.cpp.o
slurmRun: CMakeFiles/slurmRun.dir/SpikeTrain.cpp.o
slurmRun: CMakeFiles/slurmRun.dir/STTC.cpp.o
slurmRun: CMakeFiles/slurmRun.dir/build.make
slurmRun: CMakeFiles/slurmRun.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable slurmRun"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/slurmRun.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/slurmRun.dir/build: slurmRun
.PHONY : CMakeFiles/slurmRun.dir/build

CMakeFiles/slurmRun.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/slurmRun.dir/cmake_clean.cmake
.PHONY : CMakeFiles/slurmRun.dir/clean

CMakeFiles/slurmRun.dir/depend:
	cd /mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mhome/damtp/r/mw894/diss/gnm-run /mhome/damtp/r/mw894/diss/gnm-run /mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug /mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug /mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/CMakeFiles/slurmRun.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/slurmRun.dir/depend
