#!/bin/bash

# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT


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
echo "Setup the project with BasePath at $BasePath"

# Run the setup
cd "${BasePath}/ImageInference"
git submodule init
git submodule sync
git submodule update --init --recursive

# Enable conda environment
eval "$(${BasePath}/miniconda3/bin/conda shell.bash hook)"
conda activate imageinfernce

# Setup the configuartion
source config.sh