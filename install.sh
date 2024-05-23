#!/bin/bash

# Set the base path
BasePath="$(dirname -- "${BASH_SOURCE[0]}")"  # relative
BasePath="$(cd -- "$BasePath" && pwd)"  # absolutized and normalized
BasePath="$(dirname -- "$BasePath")" # go up one directory
echo "Installing needed software and dependencies in $BasePath"

# Installing Miniconda
cd "${BasePath}"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -f
rm Miniconda3-latest-Linux-x86_64.sh

# Update submodules
cd ImageInference
git submodule init
git submodule sync
git submodule update --init --recursive

# Create conda environment
conda create -yn imageinfernce python=3.10.0
eval "$(conda shell.bash hook)"
conda activate imageinfernce
./submodules/executorch/install_requirements.sh

# Fix excutorch installation which has some missing modules
cp submodules/executorch/backends/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -n -r
cp submodules/executorch/examples/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -n -r
conda install -y numpy
ulimit -n 4096

# Install Flatc
export PATH="${BasePath}/submodules/executorch/third-party/flatbuffers/cmake-out:${PATH}"
./build/install_flatc.sh

# Install buck2
if [ ! -f /tmp/buck2 ]; then
    pip3 install zstd
    wget https://github.com/facebook/buck2/releases/download/2023-07-18/buck2-x86_64-unknown-linux-musl.zst
    rm /tmp/buck2
    zstd -cdq buck2-x86_64-unknown-linux-musl.zst > /tmp/buck2 && chmod +x /tmp/buck2
    rm buck2-x86_64-unknown-linux-musl.zst
fi

# Download Android 
cd "${BasePath}"
if [ ! -f "${BasePath}/android/ndk/android-ndk-r26d" ]; then
    if [ ! -f "${BasePath}/android-ndk-r26d-linux.zip" ]; then
        rm android-ndk-r26d-linux.zip
    fi
    wget https://dl.google.com/android/repository/android-ndk-r26d-linux.zip
    mkdir android
    mkdir android/ndk
    unzip -o android-ndk-r26d-linux.zip -d android/ndk
    rm android-ndk-r26d-linux.zip
fi

./config.sh