# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_COMMAND = /home/hussain/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/hussain/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hussain/computer_vision/CourseWork/Project3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hussain/computer_vision/CourseWork/Project3/build

# Include any dependencies generated for this target.
include CMakeFiles/img.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/img.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/img.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/img.dir/flags.make

CMakeFiles/img.dir/src/image_testing.cpp.o: CMakeFiles/img.dir/flags.make
CMakeFiles/img.dir/src/image_testing.cpp.o: /home/hussain/computer_vision/CourseWork/Project3/src/image_testing.cpp
CMakeFiles/img.dir/src/image_testing.cpp.o: CMakeFiles/img.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hussain/computer_vision/CourseWork/Project3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/img.dir/src/image_testing.cpp.o"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/img.dir/src/image_testing.cpp.o -MF CMakeFiles/img.dir/src/image_testing.cpp.o.d -o CMakeFiles/img.dir/src/image_testing.cpp.o -c /home/hussain/computer_vision/CourseWork/Project3/src/image_testing.cpp

CMakeFiles/img.dir/src/image_testing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/img.dir/src/image_testing.cpp.i"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hussain/computer_vision/CourseWork/Project3/src/image_testing.cpp > CMakeFiles/img.dir/src/image_testing.cpp.i

CMakeFiles/img.dir/src/image_testing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/img.dir/src/image_testing.cpp.s"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hussain/computer_vision/CourseWork/Project3/src/image_testing.cpp -o CMakeFiles/img.dir/src/image_testing.cpp.s

CMakeFiles/img.dir/src/function_implement.cpp.o: CMakeFiles/img.dir/flags.make
CMakeFiles/img.dir/src/function_implement.cpp.o: /home/hussain/computer_vision/CourseWork/Project3/src/function_implement.cpp
CMakeFiles/img.dir/src/function_implement.cpp.o: CMakeFiles/img.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hussain/computer_vision/CourseWork/Project3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/img.dir/src/function_implement.cpp.o"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/img.dir/src/function_implement.cpp.o -MF CMakeFiles/img.dir/src/function_implement.cpp.o.d -o CMakeFiles/img.dir/src/function_implement.cpp.o -c /home/hussain/computer_vision/CourseWork/Project3/src/function_implement.cpp

CMakeFiles/img.dir/src/function_implement.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/img.dir/src/function_implement.cpp.i"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hussain/computer_vision/CourseWork/Project3/src/function_implement.cpp > CMakeFiles/img.dir/src/function_implement.cpp.i

CMakeFiles/img.dir/src/function_implement.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/img.dir/src/function_implement.cpp.s"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hussain/computer_vision/CourseWork/Project3/src/function_implement.cpp -o CMakeFiles/img.dir/src/function_implement.cpp.s

CMakeFiles/img.dir/src/csv_util.cpp.o: CMakeFiles/img.dir/flags.make
CMakeFiles/img.dir/src/csv_util.cpp.o: /home/hussain/computer_vision/CourseWork/Project3/src/csv_util.cpp
CMakeFiles/img.dir/src/csv_util.cpp.o: CMakeFiles/img.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hussain/computer_vision/CourseWork/Project3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/img.dir/src/csv_util.cpp.o"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/img.dir/src/csv_util.cpp.o -MF CMakeFiles/img.dir/src/csv_util.cpp.o.d -o CMakeFiles/img.dir/src/csv_util.cpp.o -c /home/hussain/computer_vision/CourseWork/Project3/src/csv_util.cpp

CMakeFiles/img.dir/src/csv_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/img.dir/src/csv_util.cpp.i"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hussain/computer_vision/CourseWork/Project3/src/csv_util.cpp > CMakeFiles/img.dir/src/csv_util.cpp.i

CMakeFiles/img.dir/src/csv_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/img.dir/src/csv_util.cpp.s"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hussain/computer_vision/CourseWork/Project3/src/csv_util.cpp -o CMakeFiles/img.dir/src/csv_util.cpp.s

# Object files for target img
img_OBJECTS = \
"CMakeFiles/img.dir/src/image_testing.cpp.o" \
"CMakeFiles/img.dir/src/function_implement.cpp.o" \
"CMakeFiles/img.dir/src/csv_util.cpp.o"

# External object files for target img
img_EXTERNAL_OBJECTS =

img: CMakeFiles/img.dir/src/image_testing.cpp.o
img: CMakeFiles/img.dir/src/function_implement.cpp.o
img: CMakeFiles/img.dir/src/csv_util.cpp.o
img: CMakeFiles/img.dir/build.make
img: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
img: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
img: CMakeFiles/img.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hussain/computer_vision/CourseWork/Project3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable img"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/img.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/img.dir/build: img
.PHONY : CMakeFiles/img.dir/build

CMakeFiles/img.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/img.dir/cmake_clean.cmake
.PHONY : CMakeFiles/img.dir/clean

CMakeFiles/img.dir/depend:
	cd /home/hussain/computer_vision/CourseWork/Project3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hussain/computer_vision/CourseWork/Project3 /home/hussain/computer_vision/CourseWork/Project3 /home/hussain/computer_vision/CourseWork/Project3/build /home/hussain/computer_vision/CourseWork/Project3/build /home/hussain/computer_vision/CourseWork/Project3/build/CMakeFiles/img.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/img.dir/depend

