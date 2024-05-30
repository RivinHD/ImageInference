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
cp backends/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -n -r
cp examples/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -n -r
conda install -y numpy
conda install -y scipy
ulimit -n 4096

# Install Flatc
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

cd ImageInference
source config.sh

cd submodules/executorch
./backends/qualcomm/scripts/build.sh
cp backends/ "${BasePath}/miniconda3/envs/imageinfernce/lib/python3.10/site-packages/executorch/" -n -r

# Build the requiered run time liberaries

cd "${BasePath}"/ImageInference
source config.sh