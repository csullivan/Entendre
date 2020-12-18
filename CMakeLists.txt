cmake_minimum_required(VERSION 3.2)
project(entendre C CXX)

set(USE_CUDA ON)


include(cmake/utils/FindCUDA.cmake)

set(WARNING_FLAG -Wall)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message(STATUS "Build in Debug mode")
  set(CMAKE_C_FLAGS "-O0 -g ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O0 -g ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
else()
  set(CMAKE_C_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
endif ()

# CUDA Module
find_cuda(${USE_CUDA})

if(CUDA_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

if(USE_CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif()
  message(STATUS "Build with CUDA support")

  file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})
  list(APPEND COMPILER_SRCS src/target/opt/build_cuda_on.cc)

  #list(APPEND ENTENDRE_LIBS ${CUDA_NVRTC_LIBRARY})
  list(APPEND ENTENDRE_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND ENTENDRE_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND ENTENDRE_LIBS ${CUDA_NVRTC_LIBRARY})
endif(USE_CUDA)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++14" SUPPORT_CXX14)
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_STANDARD 14)

file(GLOB ENTENDRE_SRC ${PROJECT_SOURCE_DIR}/libEntendre/src/*.cc)
add_library(entendre SHARED ${ENTENDRE_SRC})
target_include_directories(entendre PUBLIC ${PROJECT_SOURCE_DIR}/libEntendre/include)
