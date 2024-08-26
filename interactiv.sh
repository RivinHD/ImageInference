#!/bin/bash

# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

(return 0 2>/dev/null) && sourced=1 || sourced=0

if [[ ${sourced} -eq 0 ]]; then
    echo "Please run this script sourced, as environment variables are exported."
    echo "> source interactive.sh"
    exit 1
fi

# Set the base path
ProjectPath="$(dirname -- "${BASH_SOURCE[0]:-$0}")"  # relative
ProjectPath="$(cd -- "$ProjectPath" &> /dev/null && pwd 2> /dev/null)"  # absolutized and normalized

echo "Welcome to the ImageInference interactive build helper"

if [[ -z "${SETUP_DONE}" ]]; then
    source setup.sh
fi

cd "$ProjectPath"
python scripts/interactive.py
