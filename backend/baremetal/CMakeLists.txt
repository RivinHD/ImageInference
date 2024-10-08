# Original Copyright under:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Modified with the following license:
# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.19)
project(ImageInference)

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED on)

set(CMAKE_VERBOSE_MAKEFILE off)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
    set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../submodules/executorch)
endif()

if(NOT FASTOR_ROOT)
    set(FASTOR_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../submodules/fastor)
endif()

if(NOT LIBXSMM_ROOT)
    set(LIBXSMM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../submodules/libxsmm)
endif()

if(NOT CATCH2_ROOT)
    set(CATCH2_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../submodules/catch2)
endif()

if(NOT BENCHMARK_ROOT)
    set(BENCHMARK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../submodules/benchmark)
endif()

if(NOT LIBTORCH_ROOT)
    set(LIBTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../libs/libtorch)
endif()

if(ANDROID)
    set(LIBXSMM_ROOT ${LIBXSMM_ROOT}_android)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
    resolve_python_executable()
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC -fopenmp -Wall -Wextra -Wpedantic -D_GLIBCXX_USE_CXX11_ABI=0)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package(executorch CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
add_compile_definitions(USE_OMP) # Enable OpenMP usage in the Project.

target_include_directories(executorch INTERFACE ${_common_include_directories})

# Include Fastor with libxsmm backend for matrix matrix multiplication.
add_library(Fastor_HEADER_ONLY INTERFACE)
target_include_directories(Fastor_HEADER_ONLY INTERFACE ${FASTOR_ROOT})

add_library(libxsmm_HEADER_ONLY INTERFACE)
target_include_directories(libxsmm_HEADER_ONLY INTERFACE ${LIBXSMM_ROOT}/include INTERFACE ${LIBXSMM_ROOT}/src)
add_library(libxsmm SHARED IMPORTED)
add_compile_definitions(LIBXSMM_NOFORTRAN)
target_include_directories(libxsmm INTERFACE ${LIBXSMM_ROOT}/include INTERFACE ${LIBXSMM_ROOT}/src)
file(REAL_PATH ${LIBXSMM_ROOT}/lib/libxsmm.so LIBXSMM_LIB_PATH)
set_target_properties(libxsmm PROPERTIES IMPORTED_LOCATION ${LIBXSMM_LIB_PATH})


file(GLOB LIBXSMM_LIBS_VERSIONS_PATHS ${LIBXSMM_ROOT}/lib/libxsmm.so*)
install(FILES ${LIBXSMM_LIBS_VERSIONS_PATHS} DESTINATION lib)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

# Generate C++ bindings to register kernels into both PyTorch (for AOT)
# Executorch (for runtime).
gen_selected_ops(LIB_NAME "baremetal_ops_lib" ROOT_OPS "baremetal_ops::resnet50.out")

# Expect gen_selected_ops output file to be selected_operators.yaml
generate_bindings_for_kernels(
    LIB_NAME "baremetal_ops_lib" CUSTOM_OPS_YAML
    ${CMAKE_CURRENT_LIST_DIR}/baremetal_ops.yaml
)
message("Generated files ${gen_command_sources}")

# Prepare for C++ libraries.
file(GLOB shared_source
    ${CMAKE_CURRENT_LIST_DIR}/model/*.cpp
)

# C++ library to register custom ops into Executorch runtime.
set(kernel_sources ${shared_source}
    ${CMAKE_CURRENT_LIST_DIR}/execu_resnet50_out.cpp
)

add_library(imageinference_baremetal_kernels ${kernel_sources})
target_link_libraries(imageinference_baremetal_kernels PRIVATE executorch)

if(OpenMP_CXX_FOUND AND NOT ANDROID)
    target_link_libraries(imageinference_baremetal_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()

if(OpenMP_CXX_FOUND AND ANDROID)
    target_link_libraries(imageinference_baremetal_kernels PUBLIC -static-openmp -fopenmp)
endif()

target_link_libraries(imageinference_baremetal_kernels PUBLIC Fastor_HEADER_ONLY)
target_link_libraries(imageinference_baremetal_kernels PUBLIC libxsmm)
#target_compile_definitions(imageinference_baremetal_kernels PRIVATE LIBXSMM_AS_HEADER_ONLY)
target_compile_options(imageinference_baremetal_kernels PUBLIC ${_common_compile_options})

gen_operators_lib(
    LIB_NAME "baremetal_ops_lib" KERNEL_LIBS imageinference_baremetal_kernels DEPS executorch
)

install(TARGETS imageinference_baremetal_kernels baremetal_ops_lib DESTINATION lib)

if(NOT ANDROID)
    find_package(
        gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../third-party
    )

    # C++ library to register custom ops into PyTorch.
    gen_selected_ops(LIB_NAME "baremetal_ops_aot_lib" ROOT_OPS "baremetal_ops::resnet50.out")
    generate_bindings_for_kernels(
        LIB_NAME "baremetal_ops_aot_lib" CUSTOM_OPS_YAML
        ${CMAKE_CURRENT_LIST_DIR}/baremetal_ops.yaml
    )

    set(custom_ops_kernel_sources ${shared_source}
        ${CMAKE_CURRENT_LIST_DIR}/execu_resnet50.cpp # register baremetal_ops::resnet50
        ${CMAKE_CURRENT_LIST_DIR}/execu_resnet50_out.cpp # register baremetal_ops::resnet50.out
    )

    gen_custom_ops_aot_lib(
        LIB_NAME "baremetal_ops_aot_lib" KERNEL_SOURCES "${custom_ops_kernel_sources}"
    )

    target_link_libraries(baremetal_ops_aot_lib PUBLIC torch)
    target_link_libraries(baremetal_ops_aot_lib PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(baremetal_ops_aot_lib PUBLIC Fastor_HEADER_ONLY)
    target_link_libraries(baremetal_ops_aot_lib PUBLIC libxsmm_HEADER_ONLY)
    target_compile_definitions(baremetal_ops_aot_lib PRIVATE LIBXSMM_AS_HEADER_ONLY)
    target_include_directories(
        baremetal_ops_aot_lib PUBLIC ${_common_include_directories}
    )

    target_compile_options(baremetal_ops_aot_lib PUBLIC ${_common_compile_options})

    list(TRANSFORM _executor_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")

    add_executable(baremetal_ops_executor_runner ${_executor_runner__srcs})
    target_link_libraries(baremetal_ops_executor_runner PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(baremetal_ops_executor_runner PUBLIC Fastor_HEADER_ONLY)
    target_link_libraries(baremetal_ops_executor_runner PUBLIC libxsmm)
    target_link_libraries(baremetal_ops_executor_runner PUBLIC baremetal_ops_lib)
    target_link_libraries(baremetal_ops_executor_runner PUBLIC executorch)
    target_link_libraries(baremetal_ops_executor_runner PUBLIC gflags)
    #target_compile_definitions(baremetal_ops_executor_runner PRIVATE LIBXSMM_AS_HEADER_ONLY)
    target_compile_options(
        baremetal_ops_executor_runner PUBLIC ${_common_compile_options}
    )
    target_include_directories(baremetal_ops_executor_runner
        PUBLIC ${CMAKE_SOURCE_DIR}/model
        PUBLIC ${CMAKE_SOURCE_DIR}/types
    )

    install(TARGETS baremetal_ops_aot_lib DESTINATION lib)
    install(TARGETS baremetal_ops_executor_runner DESTINATION bin)

    # ################################
    # ######## Setup Testing #########
    # ################################
    find_package(Catch2 REQUIRED PATHS ${CATCH2_ROOT}/.build/lib/cmake/Catch2)
    find_package(benchmark REQUIRED PATHS ${BENCHMARK_ROOT}/.build/lib/cmake/benchmark)

    set(CURRENT_TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/test)

    file(GLOB shared_test_source
        ${CMAKE_CURRENT_LIST_DIR}/model/test/*.cpp
        ${CURRENT_TEST_DIR}/utils/*.cpp
    )

    file(GLOB TEST_FILES
        ${CURRENT_TEST_DIR}/model/*.cpp
        ${CURRENT_TEST_DIR}/types/*.cpp
    )

    list(APPEND TEST_FILES ${shared_source} ${shared_test_source})

    add_executable(tests ${TEST_FILES})
    target_compile_options(tests PRIVATE -g -fsanitize=address -fsanitize=leak -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(tests PRIVATE -g -fsanitize=address -fsanitize=leak -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_libraries(tests PUBLIC torch)
    target_link_libraries(tests PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(tests PUBLIC Fastor_HEADER_ONLY)
    target_link_libraries(tests PUBLIC libxsmm)
    target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
    target_compile_definitions(tests PRIVATE IMAGEINFERENCE_TESTING)
    target_compile_options(tests PUBLIC ${_common_compile_options})

    add_executable(tests_valgrind ${TEST_FILES}) # Tests without asan for valgrind usage
    target_compile_options(tests_valgrind PRIVATE -g)
    target_link_options(tests_valgrind PRIVATE -g)
    target_link_libraries(tests_valgrind PUBLIC torch)
    target_link_libraries(tests_valgrind PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(tests_valgrind PUBLIC Fastor_HEADER_ONLY)
    target_link_libraries(tests_valgrind PUBLIC libxsmm)
    target_link_libraries(tests_valgrind PRIVATE Catch2::Catch2WithMain)
    target_compile_definitions(tests_valgrind PRIVATE IMAGEINFERENCE_TESTING)
    target_compile_options(tests_valgrind PUBLIC ${_common_compile_options})

    file(GLOB BENCHMARK_FILES
        ${CURRENT_TEST_DIR}/benchmarks/*.cpp
    )

    list(APPEND BENCHMARK_FILES ${shared_source})

    add_executable(benchmarks ${BENCHMARK_FILES})
    target_compile_options(tests PRIVATE -g)
    target_link_options(tests PRIVATE -g)
    target_link_libraries(benchmarks PUBLIC benchmark::benchmark)
    target_link_libraries(benchmarks PUBLIC benchmark::benchmark_main)
    target_link_libraries(benchmarks PUBLIC torch)
    target_link_libraries(benchmarks PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(benchmarks PUBLIC Fastor_HEADER_ONLY)
    target_link_libraries(benchmarks PUBLIC libxsmm)
    target_compile_definitions(benchmarks PRIVATE IMAGEINFERENCE_BENCHMARK)
    target_compile_options(benchmarks PUBLIC ${_common_compile_options})
    target_link_libraries(benchmarks PUBLIC benchmark::benchmark)
    target_link_libraries(benchmarks PUBLIC benchmark::benchmark_main)

    include(CTest)
    include(Catch)

    enable_testing()
    catch_discover_tests(tests)
endif()