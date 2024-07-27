# Original Copyright under:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Modified with the following license:
# Copyright (c) 2024 by Vincent Gerlach. All rights reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# in the root folder of this project with the name LICENSE. If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.19)
project(ImageInference)

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
    set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../submodules/executorch)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
    resolve_python_executable()
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package(executorch CONFIG REQUIRED)

target_include_directories(executorch INTERFACE ${_common_include_directories})

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
file(GLOB shared_shource
    ${CMAKE_CURRENT_LIST_DIR}/model/*.cpp
)

# C++ library to register custom ops into Executorch runtime.
set(kernel_sources ${shared_shource}
    ${CMAKE_CURRENT_LIST_DIR}/execu_resnet50_out.cpp
)

add_library(imageinference_baremetal_kernels ${kernel_sources})
target_link_libraries(imageinference_baremetal_kernels PRIVATE executorch)
target_compile_options(imageinference_baremetal_kernels PUBLIC ${_common_compile_options})

gen_operators_lib(
    LIB_NAME "baremetal_ops_lib" KERNEL_LIBS imageinference_baremetal_kernels DEPS executorch
)

install(TARGETS imageinference_baremetal_kernels baremetal_ops_lib DESTINATION lib)

if(NOT ANDROID)
    # C++ library to register custom ops into PyTorch.
    gen_selected_ops(LIB_NAME "baremetal_ops_aot_lib" ROOT_OPS "baremetal_ops::resnet50.out")
    generate_bindings_for_kernels(
        LIB_NAME "baremetal_ops_aot_lib" CUSTOM_OPS_YAML
        ${CMAKE_CURRENT_LIST_DIR}/baremetal_ops.yaml
    )

    set(custom_ops_kernel_sources ${shared_shource}
        ${CMAKE_CURRENT_LIST_DIR}/execu_resnet50.cpp # register baremetal_ops::resnet50
        ${CMAKE_CURRENT_LIST_DIR}/execu_resnet50_out.cpp # register baremetal_ops::resnet50.out
    )

    gen_custom_ops_aot_lib(
        LIB_NAME "baremetal_ops_aot_lib" KERNEL_SOURCES "${custom_ops_kernel_sources}"
    )

    target_include_directories(
        baremetal_ops_aot_lib PUBLIC ${_common_include_directories}
    )

    target_compile_options(baremetal_ops_aot_lib PUBLIC ${_common_compile_options})

    install(TARGETS baremetal_ops_aot_lib DESTINATION lib)
endif()
