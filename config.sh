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

# DO NOT EDIT THE CODE BELOW
# Check sourced
(return 0 2>/dev/null) && sourced=1 || sourced=0

if [[ ${sourced} -eq 0 ]]; then
    echo "Please run this script sourced, as environment variables are exported."
    echo "> source config.sh"
    exit 1
fi

# Set the base path
BasePath="$(dirname -- "${BASH_SOURCE[0]}")"  # relative
BasePath="$(cd -- "$BasePath" && pwd)"  # absolutized and normalized
BasePath="$(dirname -- "$BasePath")" # go up one directory

# ==============================================================================
# YOU CAN ADJUST THE VALUE OF THE VARIABLES BELOW

# Export the required environment variables
export ANDROID_NDK="${BasePath}/android/ndk/android-ndk-r26d"
export ANDROID_ABI="arm64-v8a" # Options: arm64-v8a, armeabi-v7a, x86, x86_64 but currently excutorch only supports arm64-v8a.
export ANDROID_VERSION="29"
export QNN_SDK_ROOT="/opt/qcom/aistack/qairt/2.21.0.240401"
export IMAGENET_DATASET_2012="${BasePath}/data/imagenet/ilsvrc_2012"

# DO NOT EDIT ANY CODE OUTSIDE THIS SECTION
# ==============================================================================

# DO NOT EDIT THE CODE BELOW

export FLATC_EXECUTABLE="$(which -- flatc)"
export LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang/":$LD_LIBRARY_PATH
export EXECUTORCH_ROOT="${BasePath}/ImageInference/submodules/executorch"
echo "ANDROID_NDK: $ANDROID_NDK"
echo "ANDROID_ABI: $ANDROID_ABI"
echo "ANDROID_VERSION: $ANDROID_VERSION"
echo "QNN_SDK_ROOT: $QNN_SDK_ROOT"
echo "IMAGENET_DATASET_2012: $IMAGENET_DATASET_2012"
echo "FLATC_EXECUTABLE: $FLATC_EXECUTABLE"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "EXECUTORCH_ROOT: $EXECUTORCH_ROOT"

export SETUP_DONE=1  # Mark the setup as done