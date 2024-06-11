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
./install_requirements.sh

# Fix excutorch installation which has some missing modules
yes | cp backends/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -r
yes | cp examples/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -r
conda install -y numpy
conda install -y scipy
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
yes | cp backends/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -r

# Number of available processors
if [ "$(uname)" == "Darwin" ]; then
  CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
  CMAKE_JOBS=$(( $(nproc) - 1 ))
fi

# Build the requiered run time liberary
rm -rf cmake-android-out
mkdir cmake-android-out
cmake . -DCMAKE_INSTALL_PREFIX=cmake-android-out \
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
    -DFLATC_EXECUTABLE="${FLATC_EXECUTABLE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -Bcmake-android-out
    # For now Vulkan does not work properly and therefore is disabled
    # -DEXECUTORCH_BUILD_VULKAN=ON \

cmake --build cmake-android-out -j "${CMAKE_JOBS}" --target install

# Build the android extension
cmake extension/android \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DCMAKE_INSTALL_PREFIX=cmake-android-out \
  -DANDROID_PLATFORM="${ANDROID_VERSION}" \
  -DCMAKE_BUILD_TYPE=Release \
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

# Print the config for user verfication
cd "${BasePath}/ImageInference"
source config.sh