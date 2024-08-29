#!/bin/bash

# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

# Check sourced
(return 0 2>/dev/null) && sourced=1 || sourced=0

if [[ ${sourced} -eq 0 ]]; then
    echo "Please run this script sourced, as environment variables are exported."
    echo "> source install.sh"
    exit 1
fi

# Set the base path
BasePath="$(dirname -- "${BASH_SOURCE[0]:-$0}")"  # relative
BasePath="$(cd -- "$BasePath" &> /dev/null && pwd 2> /dev/null)"  # absolutized and normalized
BasePath="$(dirname -- "$BasePath")" # go up one directory
echo "Installing needed software and dependencies in $BasePath"

# Ask the user if they want to continue
read -p "Do you want to continue the installation process? (y/n): " answer
case ${answer:0:1} in
    y|Y )
        echo "Continuing the installation process..."
    ;;
    * )
        echo "Installation process aborted."
        return 1 2>/dev/null # Exit bash script in source and none-source mode
        exit 1
    ;;
esac

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
eval "$(${BasePath}/miniconda3/bin/conda shell.bash hook)"
conda create -yn imageinfernce python=3.10.0
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
if [ $? -ne 0 ]; then
  echo "install_requirements.sh failed, retrying..."
  if [ -f "pip-out/temp.linux-x86_64-cpython-310/cmake-out/buck2-bin/buck2-3bbde7daa94987db468d021ad625bc93dc62ba7fcb16945cb09b64aab077f284" ]; then
      pip-out/temp.linux-x86_64-cpython-310/cmake-out/buck2-bin/buck2-3bbde7daa94987db468d021ad625bc93dc62ba7fcb16945cb09b64aab077f284 clean
  fi
  ./install_requirements.sh
fi

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
cd "${BasePath}/ImageInference"
source config.sh

# Number of available processors
if [ "$(uname)" == "Darwin" ]; then
  CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
  CMAKE_JOBS=$(( $(nproc) - 1 ))
fi

# Install libxsmm
cd "${BasePath}/ImageInference/submodules/libxsmm"
make -j "${CMAKE_JOBS}" STATIC=0 \
  BLAS=0\
  LIBXSMM_NO_BLAS=1

# Install libxsmm for android
cd "${BasePath}/ImageInference/submodules/libxsmm_android"

sed -i 's/libxsmm_cpuid_arm.c libxsmm_cpuid_x86.c libxsmm_generator.c libxsmm_trace.c libxsmm_matrixeqn.c)/libxsmm_cpuid_arm.c libxsmm_cpuid_x86.c libxsmm_generator.c libxsmm_matrixeqn.c)/' Makefile
sed -i 's/LIBCPP := $(call ldclib,$(LD),$(SLDFLAGS),stdc++)/LIBCPP := $(call ldclib,$(LD),$(SLDFLAGS))/' Makefile.inc
sed -i '/^# define LIBXSMM_TRACE$/s/^/\/\//' src/libxsmm_trace.h

# Empty target to build with compiler default
case "$ANDROID_ABI" in
  arm64-v8a)
    MAPPED_ABI="aarch64"
    ;;
  armeabi-v7a)
    MAPPED_ABI="armv7a"
    ;;
  x86)
    MAPPED_ABI="i686"
    ;;
  x86_64)
    MAPPED_ABI="x86_64"
    ;;
  *)
    echo "!!!!!!!!!!!!!!! Unknown architecture: ${ANDROID_ABI} !!!!!!!!!!!!!!!"
    ;;
esac

ANDROID_COMPILER_DIR="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin"
ANDROID_COMPILER="${ANDROID_COMPILER_DIR}/${MAPPED_ABI}-linux-android${ANDROID_VERSION}-clang"
make -j "${CMAKE_JOBS}" \
  STATIC=0 \
  WERROR=0 \
  CC="${ANDROID_COMPILER}" \
  LD="${ANDROID_COMPILER}++" \
  CXX="${ANDROID_COMPILER}++" \
  FORTRAN=0 \
  BLAS=0 \
  TRACE=0 \
  LIBXSMM_NO_BLAS=1 \
  TARGET= \
  RPM_OPT_FLAGS="--target=aarch64-none-linux-android${ANDROID_VERSION} -DANDROID"
  

# Install catch2
cd "${BasePath}/ImageInference/submodules/catch2"
cmake . \
  -DCMAKE_INSTALL_PREFIX=.build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -B.build 
cmake --build .build -j "${CMAKE_JOBS}" --target install

# Install benchmark
cd "${BasePath}/ImageInference/submodules/benchmark"
if ! grep -q 'add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)' CMakeLists.txt; then
  sed -i '/if (MSVC)/i add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)' CMakeLists.txt
fi
cmake . \
  -DCMAKE_INSTALL_PREFIX=.build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON \
  -DCMAKE_CXX_STANDARD=17 \
  -B.build 
cmake --build .build -j "${CMAKE_JOBS}" --target install

# Setup the qualcomm backend
cd "${BasePath}/ImageInference/submodules/executorch"
# Workaround for fbs files in exir/_serialize
yes | cp schema/program.fbs exir/_serialize/program.fbs
yes | cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs
./backends/qualcomm/scripts/build.sh
yes | cp backends/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -r &> /dev/null

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
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -Bcmake-android-out
    # For now Vulkan does not work properly and therefore is disabled
    # -DEXECUTORCH_BUILD_VULKAN=ON \
    # FIXME: Do no forget to disable logging i.e. -DEXECUTORCH_ENABLE_LOGGING=OFF

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

cd "${BasePath}/ImageInference/submodules/executorch"

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

# Register the custome operater into the android CMakeLists.txt
if ! grep -q 'if(EXECUTORCH_BUILD_IMAGEINFERENCE_BAREMETAL)' extension/android/CMakeLists.txt; then
    sed -i '/add_library(executorch_jni SHARED jni\/jni_layer.cpp)/i \
option(EXECUTORCH_BUILD_IMAGEINFERENCE_BAREMETAL "Include the imageinference baremetal custome kernels" OFF)\
if(EXECUTORCH_BUILD_IMAGEINFERENCE_BAREMETAL)\
  message(STATUS "Register imageinference baremetal kernels")\
  if(NOT FASTOR_ROOT)\
    set(FASTOR_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../fastor)\
  endif()\
  if(NOT LIBXSMM_ROOT)\
    set(LIBXSMM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../libxsmm_android)\
  endif()\
  add_library(imageinference_baremetal_kernels STATIC IMPORTED)\
  set_property(TARGET imageinference_baremetal_kernels PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/../../lib/libimageinference_baremetal_kernels.a)\
  add_library(baremetal_ops_lib STATIC IMPORTED)\
  set_property(TARGET baremetal_ops_lib PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/../../lib/libbaremetal_ops_lib.a)\
  find_package(OpenMP REQUIRED)\
  add_library(Fastor_HEADER_ONLY INTERFACE)\
  target_include_directories(Fastor_HEADER_ONLY INTERFACE ${FASTOR_ROOT})\
  add_library(libxsmm SHARED IMPORTED)\
  add_compile_definitions(LIBXSMM_NOFORTRAN)\
  target_include_directories(libxsmm INTERFACE ${LIBXSMM_ROOT}/include INTERFACE ${LIBXSMM_ROOT}/src)\
  set_target_properties(libxsmm PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/../../lib/libxsmm.so)\
  list(APPEND link_libraries baremetal_ops_lib imageinference_baremetal_kernels -static-openmp -fopenmp Fastor_HEADER_ONLY libxsmm)\
  target_link_options_shared_lib(baremetal_ops_lib)\
endif()\n' extension/android/CMakeLists.txt
fi

# Register dependencies into the android app
if ! grep -q 'NativeLoader.loadLibrary("xsmm")' extension/android/src/main/java/org/pytorch/executorch/NativePeer.java; then
    sed -i '/\/\/ Loads libexecutorch.so from jniLibs/i \
    // Load dependencies of libexecutorch.so\
    NativeLoader.loadLibrary("c");\
    NativeLoader.loadLibrary("m");\
    NativeLoader.loadLibrary("dl");\
    NativeLoader.loadLibrary("c++_shared");\
    NativeLoader.loadLibrary("xsmm");\n' extension/android/src/main/java/org/pytorch/executorch/NativePeer.java
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
yes | cp "cmake-android-out/lib/libxsmm.so" \
   "${BasePath}/ImageInference/android/app/src/main/jniLibs/${ANDROID_ABI}/libxsmm.so"
# yes | cp "cmake-android-out/lib/libm.so"* \
#    "${BasePath}/ImageInference/android/app/src/main/jniLibs/${ANDROID_ABI}"
# yes | cp "cmake-android-out/lib/libc.so"* \
#    "${BasePath}/ImageInference/android/app/src/main/jniLibs/${ANDROID_ABI}"
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
cd "${BasePath}/ImageInference/submodules/executorch"

if ! grep -q 'add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)' CMakeLists.txt; then
  sed -i '/option(EXECUTORCH_ENABLE_LOGGING "Build with ET_LOG_ENABLED"/i \
if(USE_OLD_ABI)\
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)\
endif()\n' CMakeLists.txt
fi

rm -rf cmake-out
cmake \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DUSE_OLD_ABI=ON \
    -Bcmake-out .
cmake --build cmake-out -j "${CMAKE_JOBS}" --target install --config Release

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

cd "${BasePath}/ImageInference"
python "${BasePath}/ImageInference/scripts/export_resnet50_for_test.py"

cd "${BasePath}/ImageInference/submodules/executorch"
cd "$BUILD_DIR"
ctest --output-on-failure -C Release

python "${BasePath}/ImageInference/scripts/copy_imagenet_2012.py"

# Print the config for user verfication
cd "${BasePath}/ImageInference"
source config.sh