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
BasePath="$(dirname -- "${BASH_SOURCE[0]:-$0}")"  # relative
BasePath="$(cd -- "$BasePath" &> /dev/null && pwd 2> /dev/null)"  # absolutized and normalized
BasePath="$(dirname -- "$BasePath")" # go up one directory
BasePath="$(dirname -- "$BasePath")" # go up one directory
BasePath="$(dirname -- "$BasePath")" # go up one directory
echo "Setup the project with BasePath at $BasePath"

# Installing Miniconda
cd "${BasePath}"
if [ ! -d "${BasePath}/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -f
    rm Miniconda3-latest-Linux-x86_64.sh
fi

# Create conda environment
eval "$(${BasePath}/miniconda3/bin/conda shell.bash hook)"
conda create -yn imageinfernceAimetQuantize python=3.10.0
conda activate imageinfernceAimetQuantize

pip install aimet-torch


# Setup the configuartion
source ${BasePath}/ImageInference/config.sh