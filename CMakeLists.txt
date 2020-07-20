#Do this command 
#$ sudo cmake CMakeLists.txt
#$ sudo make
#$ sudo ./pong
cmake_minimum_required(VERSION 2.8)
project(pong)
set(CMAKE_CXX_FLAGS "-O3")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(SOURCES main.cpp pinball_game.hpp)
add_executable(pong ${SOURCES})
target_link_libraries(pong ${OpenCV_LIBS})
