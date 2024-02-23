cmake_minimum_required(VERSION 2.8)
project(ObjectRecognition) # Avoid spaces in project names
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(vidDisplay src/thresholding.cpp header_files/objfun.h src/function_implement.cpp header_files/csv_util.h src/csv_util.cpp)
add_executable(img src/image_testing.cpp header_files/objfun.h src/function_implement.cpp header_files/csv_util.h src/csv_util.cpp)
add_executable(try src/try.cpp header_files/objfun.h src/function_implement.cpp header_files/csv_util.h src/csv_util.cpp)
target_link_libraries(vidDisplay ${OpenCV_LIBS})
target_link_libraries(img ${OpenCV_LIBS})
target_link_libraries(try ${OpenCV_LIBS})