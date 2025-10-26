cmake_minimum_required(VERSION 3.18)
project(GNN_Traffic_Pred CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# --- LibTorch must be handled manually ---
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip")
set(LIBTORCH_ROOT "${CMAKE_BINARY_DIR}/libtorch")

if(NOT EXISTS ${LIBTORCH_ROOT})
  message(STATUS "Downloading LibTorch...")
  file(DOWNLOAD ${LIBTORCH_URL} ${CMAKE_BINARY_DIR}/libtorch.zip SHOW_PROGRESS)
  file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/libtorch.zip DESTINATION ${CMAKE_BINARY_DIR})
endif()


list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")


find_package(Torch REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")


include_directories(include)
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(gnn ${SOURCES})

target_link_libraries(
  gnn
  PRIVATE
  ${TORCH_LIBRARIES}
  nlohmann_json::nlohmann_json
)

set_property(TARGET gnn PROPERTY CXX_STANDARD 17)
set_property(TARGET gnn PROPERTY POSITION_INDEPENDENT_CODE ON)

target_precompile_headers(gnn PRIVATE
    <torch/torch.h>
    <nlohmann/json.hpp>
)

