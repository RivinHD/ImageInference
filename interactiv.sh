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
