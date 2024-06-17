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
    echo "> source setup.sh"
    exit 1
fi

# Set the base path
BasePath="$(dirname -- "${BASH_SOURCE[0]}")"  # relative
BasePath="$(cd -- "$BasePath" && pwd)"  # absolutized and normalized
BasePath="$(dirname -- "$BasePath")" # go up one directory
echo "Setup the project with BasePath at $BasePath"

# Run the setup
cd "${BasePath}/ImageInference"
git submodule init
git submodule sync
git submodule update --init --recursive

# Enable conda environment
eval "$(conda shell.bash hook)"
conda activate imageinfernce

# Setup the configuartion
source config.sh