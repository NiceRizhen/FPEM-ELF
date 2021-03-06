CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(atari)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")


# add elf and vendor
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../elf/ ${CMAKE_BINARY_DIR}/elf/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/ ${CMAKE_BINARY_DIR}/vendor/)

# add ALE
find_package(PkgConfig)
pkg_check_modules(ALE REQUIRED ale)
include_directories(SYSTEM ${ALE_INCLUDE_DIRS})
link_directories(${ALE_LIBRARY_DIRS})


# the python lib
file(GLOB SOURCES *.cc)
pybind11_add_module(atari_game ${SOURCES})
target_link_libraries(atari_game PRIVATE elf ale)
set_target_properties(atari_game
	PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")


add_executable(benchmark-ale benchmark-ale.cpp)
target_link_libraries(benchmark-ale ale concurrentqueue)

add_executable(benchmark-atari benchmark-atari.cpp atari_game.cc)
target_link_libraries(benchmark-atari ale elf pybind11)
