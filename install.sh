#!/bin/bash

#  Copyright (c) 2024 by Vincent Gerlach. All rights reserved.
#
#  SPDX-License-Identifier: GPL-3.0-or-later
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  in the root folder of this project with the name LICENSE. If not, see <http://www.gnu.org/licenses/>.

# Check sourced
(return 0 2>/dev/null) && sourced=1 || sourced=0

if [[ ${sourced} -eq 0 ]]; then
    echo "Please run this script sourced, as environment variables are exported."
    echo "> source install.sh"
    exit 1
fi

# Set the base path
BasePath="$(dirname -- "${BASH_SOURCE[0]}")"  # relative
BasePath="$(cd -- "$BasePath" && pwd)"  # absolutized and normalized
BasePath="$(dirname -- "$BasePath")" # go up one directory
echo "Installing needed software and dependencies in $BasePath"

# Installing Miniconda
cd "${BasePath}"
if [ ! -d "${BasePath}/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -f
    rm Miniconda3-latest-Linux-x86_64.sh
fi

# Update submodules
cd ImageInference
git submodule init
git submodule sync
git submodule update --init --recursive


# Create conda environment
conda create -yn imageinfernce python=3.10.0
eval "$(conda shell.bash hook)"
conda activate imageinfernce
cd submodules/executorch
if [ -f "pip-out/temp.linux-x86_64-cpython-310/cmake-out/buck2-bin/buck2-3bbde7daa94987db468d021ad625bc93dc62ba7fcb16945cb09b64aab077f284" ]; then
    pip-out/temp.linux-x86_64-cpython-310/cmake-out/buck2-bin/buck2-3bbde7daa94987db468d021ad625bc93dc62ba7fcb16945cb09b64aab077f284 clean
fi

# Update the executorch install_requierements with existing pytorch versions
sed -i 's/NIGHTLY_VERSION=dev20240507/NIGHTLY_VERSION=dev20240716/' install_requirements.sh
sed -i 's/torch=="2.4.0.${NIGHTLY_VERSION}"/torch=="2.5.0.${NIGHTLY_VERSION}"/' install_requirements.sh
sed -i 's/torchvision=="0.19.0.${NIGHTLY_VERSION}"/torchvision=="0.20.0.${NIGHTLY_VERSION}"/' install_requirements.sh
sed -i 's/torchaudio=="2.2.0.${NIGHTLY_VERSION}"/torchaudio=="2.4.0.${NIGHTLY_VERSION}"/' install_requirements.sh

# Install the requiered software for executorch
./install_requirements.sh

# Fix excutorch installation which has some missing modules
yes | cp backends/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -r &> /dev/null
yes | cp examples/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -r &> /dev/null
pip install numpy==2.0.1
pip install scipy
ulimit -n 4096

# Install Flatc
./build/install_flatc.sh
export PATH="$(pwd)/third-party/flatbuffers/cmake-out:${PATH}"
./build/install_flatc.sh

# Download Android 
cd "${BasePath}"
if [ ! -d "${BasePath}/android/ndk/android-ndk-r26d" ]; then
    if [ ! -f "${BasePath}/android-ndk-r26d-linux.zip" ]; then
        rm android-ndk-r26d-linux.zip
    fi
    wget https://dl.google.com/android/repository/android-ndk-r26d-linux.zip
    mkdir android
    mkdir android/ndk
    unzip -o android-ndk-r26d-linux.zip -d android/ndk
    rm android-ndk-r26d-linux.zip
fi

# Initalize the config variables needed for the rest of the installation
cd ImageInference
source config.sh

# Setup the qualcomm backend
cd submodules/executorch
# Workaround for fbs files in exir/_serialize
yes | cp schema/program.fbs exir/_serialize/program.fbs
yes | cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs
./backends/qualcomm/scripts/build.sh
yes | cp backends/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -r &> /dev/null

# Number of available processors
if [ "$(uname)" == "Darwin" ]; then
  CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
  CMAKE_JOBS=$(( $(nproc) - 1 ))
fi

# Build the requiered run time liberary
rm -rf cmake-android-out
mkdir cmake-android-out
cmake . \
    -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DANDROID_NATIVE_API_LEVEL="${ANDROID_VERSION}" \
    -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT="${QNN_SDK_ROOT}" \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DFLATC_EXECUTABLE="${FLATC_EXECUTABLE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -Bcmake-android-out
    # For now Vulkan does not work properly and therefore is disabled
    # -DEXECUTORCH_BUILD_VULKAN=ON \

cmake --build cmake-android-out -j "${CMAKE_JOBS}" --target install

# Building the custome ops in backend/baremetal for the android application
if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

SITE_PACKAGES="$(${PYTHON_EXECUTABLE} -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
CMAKE_INSTALL_PATH="${BasePath}/ImageInference/submodules/executorch/cmake-android-out"
CMAKE_PREFIX_PATH="${CMAKE_INSTALL_PATH}/lib/cmake/ExecuTorch;${SITE_PACKAGES}/torch/share/cmake/Torch;"
BUILD_DIR="cmake-android-out/portable/custom_ops"

cd submodules/executorch

rm -rf "$BUILD_DIR"
cmake \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DCMAKE_BUILD_TYPE=Release \
    -B"$BUILD_DIR" \
    "${BasePath}/ImageInference/backend/baremetal"

cmake --build "$BUILD_DIR" -j "${CMAKE_JOBS}" --target install

# Register the libarary in the executorch-config.cmake
if ! grep -q 'if(EXECUTORCH_BUILD_IMAGEINFERENCE_BAREMETAL)' cmake-android-out/lib/cmake/ExecuTorch/executorch-config.cmake; then
    sed -i '/foreach(lib ${lib_list})/i \
option(EXECUTORCH_BUILD_IMAGEINFERENCE_BAREMETAL "Include the imageinference baremetal custome kernels" OFF)\
if(EXECUTORCH_BUILD_IMAGEINFERENCE_BAREMETAL)\
  list(APPEND lib_list baremetal_ops_lib imageinference_baremetal_kernels)\
endif()\n' cmake-android-out/lib/cmake/ExecuTorch/executorch-config.cmake
fi

# Register the custome operater into the android CMakeLists.txt
if ! grep -q 'if(EXECUTORCH_BUILD_IMAGEINFERENCE_BAREMETAL)' extension/android/CMakeLists.txt; then
    sed -i '/add_library(executorch_jni SHARED jni\/jni_layer.cpp)/i \
option(EXECUTORCH_BUILD_IMAGEINFERENCE_BAREMETAL "Include the imageinference baremetal custome kernels" OFF)\
if(EXECUTORCH_BUILD_IMAGEINFERENCE_BAREMETAL)\
  list(APPEND link_libraries baremetal_ops_lib imageinference_baremetal_kernels)\
  target_link_options_shared_lib(baremetal_ops_lib)\
  find_package(OpenMP REQUIRED) \
  list(APPEND link_libraries -static-openmp -fopenmp)\
endif()\n' extension/android/CMakeLists.txt
fi

# Build the android extension
cmake extension/android \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DCMAKE_INSTALL_PREFIX=cmake-android-out \
  -DANDROID_PLATFORM="${ANDROID_VERSION}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_IMAGEINFERENCE_BAREMETAL=ON \
  -Bcmake-android-out/extension/android

cmake --build cmake-android-out/extension/android -j "${CMAKE_JOBS}"

# Copy the needed libaraies to the android application
mkdir -p "${BasePath}/ImageInference/android/app/src/main/jniLibs/${ANDROID_ABI}"
yes | cp "cmake-android-out/extension/android/libexecutorch_jni.so" \
    "${BasePath}/ImageInference/android/app/src/main/jniLibs/${ANDROID_ABI}/libexecutorch.so"
yes | cp "cmake-android-out/lib/libqnn_executorch_backend.so" \
   "${BasePath}/ImageInference/android/app/src/main/jniLibs/${ANDROID_ABI}"
yes | cp "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so" \
    "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so" \
    "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV69Stub.so" \
    "${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so" \
    "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV73Stub.so" \
    "${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so" \
    "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV75Stub.so" \
    "${QNN_SDK_ROOT}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so" \
    "${BasePath}/ImageInference/android/app/src/main/jniLibs/${ANDROID_ABI}"

# Building the custome ops in backend/baremetal for the pytorch usage
source "${BasePath}/ImageInference/submodules/executorch/.ci/scripts/utils.sh"
cmake_install_executorch_lib

SITE_PACKAGES="$(${PYTHON_EXECUTABLE} -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
CMAKE_INSTALL_PATH="${BasePath}/ImageInference/submodules/executorch/cmake-out"
CMAKE_PREFIX_PATH="${CMAKE_INSTALL_PATH}/lib/cmake/ExecuTorch;${SITE_PACKAGES}/torch/share/cmake/Torch;${SITE_PACKAGES}/torch/lib;"
BUILD_DIR="cmake-out/portable/custom_ops"

rm -rf "$BUILD_DIR"
cmake "${BasePath}/ImageInference/backend/baremetal"\
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DCMAKE_BUILD_TYPE=Release \
    -B"$BUILD_DIR"

cmake --build "$BUILD_DIR" -j "${CMAKE_JOBS}" --target install

python "${BasePath}/ImageInference/scripts/copy_imagenet_2012.py"

# Print the config for user verfication
cd "${BasePath}/ImageInference"
source config.sh