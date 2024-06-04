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

import os
import shutil

MODELS = [
    "resnet50v1.5",
]
HTP_QUALCOMM = "HTP (Qualcomm)"
HARDWARE = [
    "CPU (XNNPACK)",
    # "GPU (VULKAN)",
    HTP_QUALCOMM,
]
QUALCOMM_DEVICES = [
    "SM8650 (Snapdragon 8 Gen 3)",
    "SM8550 (Snapdragon 8 Gen 2)",
    "SM8475 (Snapdragon 8+ Gen 1)",
    "SM8450 (Snapdragon 8 Gen 1)",
]
MODELS_TO_PYFILE = {
    "resnet50v1.5": "resnet50v15",
}
HARDWARE_TO_DIRECTORY = {
    "CPU (XNNPACK)": "xnnpack",
    "GPU (VULKAN)": "vulkan",
    HTP_QUALCOMM: "htp",
}
EXECU_PACKAGE = "backend.execu_python"


def select_models() -> list[str]:
    models = []
    while len(models) == 0:
        print("Select models to process separated by comma or 'all' to process all models.")
        print(f"Available models are {MODELS}")
        text = input("> ").strip()
        if text == 'all':
            models = MODELS
        else:
            models = [m.strip() for m in text.split(",") if m.strip() in MODELS]
    print(f"Selected models are {models}")
    return models


def ask_correct_model() -> bool:
    text = input("Are the selected model correct? (y/n) > ").strip()
    while not (text.startswith("y") or text.startswith("n")):
        text = input("Please enter y or n > ")
    return text.startswith("y")


def select_hardware() -> list[str]:
    hardware = []
    while len(hardware) == 0:
        print("Select Hardware to run on separated by comma or 'all' to support every hardware.")
        print(f"Available hardware is {HARDWARE}")
        text = input("> ").strip()
        if text == 'all':
            hardware = HARDWARE
        else:
            for h in text.split(","):
                h = h.strip().lower()
                for hw in HARDWARE:
                    if hw.lower().startswith(h):
                        hardware.append(hw)
    print(f"Selected hardware is {hardware}")
    return hardware


def ask_correct_hardware() -> bool:
    text = input("Is the selected hardware correct? (y/n) > ").strip()
    while not (text.startswith("y") or text.startswith("n")):
        text = input("Please enter y or n > ")
    return text.startswith("y")


def select_htp_devices() -> str:
    devices = []
    while len(devices) == 0:
        print("Since you want to use HTP you need to select Qualcomm device to run on.")
        print("Select devices to run on separated by comma or 'all' to support every listed devices.")
        print(f"Available devices are {QUALCOMM_DEVICES}")
        input_devices = input("> ").strip().lower().split(",")
        for device in input_devices:
            for qd in QUALCOMM_DEVICES:
                qdr = qd.split(" ")[0].lower()
                if device == qdr:
                    devices.append(qd.split(" ")[0])
                    break
    print(f"Selected device is {devices}")
    return devices


def ask_quantization() -> tuple[bool, bool]:
    text = input("Do you want to quantize the model or 'both' to get quantized and non-quantized models? (y/n/both) > ")
    text = text.strip().lower()
    while not (text.startswith("y") or text.startswith("n") or text.startswith("both")):
        text = input("Please enter y or n or both > ")
    return (not text.startswith("y"), not text.startswith("n"))


def ask_start_processing() -> bool:
    text = input("Start processing? (y/n) > ").strip().lower()
    while not (text.startswith("y") or text.startswith("n")):
        text = input("Please enter y or n > ")
    return text.startswith("y")


def _generate_quantize_command(base_command: str, non_quantize: bool, quantize: bool) -> list[str]:
    commands = []
    if quantize:
        commands.append(f"{base_command} --quantize")
    if non_quantize:
        commands.append(base_command)
    return commands


def generate_command(
        models: list[str],
        hardware: list[str],
        non_quantize: bool,
        quantize: bool,
        htp_devices: list[str] = []) -> str:
    commands = []
    for model in models:
        for hw in hardware:
            package_path = ".".join([EXECU_PACKAGE, HARDWARE_TO_DIRECTORY[hw], MODELS_TO_PYFILE[model]])
            base_command = f"python -m {package_path}"
            if hw == HTP_QUALCOMM:
                for device in htp_devices:
                    model_command = f"{base_command} --model {device}"
                    commands.extend(_generate_quantize_command(model_command, non_quantize, quantize))
            else:
                commands.extend(_generate_quantize_command(base_command, non_quantize, quantize))
    return "; ".join(commands)


def ask_copy_to_assets() -> bool:
    text = input("Should the generated models by copied to the android app? (y/n) > ").strip().lower()
    while not (text.startswith("y") or text.startswith("n")):
        text = input("Please enter y or n > ")
    return text.startswith("y")


if __name__ == "__main__":

    # Ask for models
    models = select_models()
    models_correct = ask_correct_model()
    while not models_correct:
        models = select_models()
        models_correct = ask_correct_model()
    print()

    # Ask for hardware
    hardware = select_hardware()
    hardware_correct = ask_correct_hardware()
    while not hardware_correct:
        hardware = select_hardware()
        hardware_correct = ask_correct_hardware()
    print()

    htp_devices = ""
    if HTP_QUALCOMM in hardware:
        htp_devices = select_htp_devices()
        print()

    # Ask for quantization
    non_quantize, quantize = ask_quantization()
    print()

    # Generate command and start processing
    command = generate_command(models, hardware, non_quantize, quantize, htp_devices)
    print("The command that will be executed is:")
    print("-"*50)
    print(command)
    print("-"*50)
    working_directory = os.path.dirname(os.path.dirname(__file__))
    os.chdir(working_directory)
    print(f"In the working directory: {working_directory}")

    start = ask_start_processing()
    if not start:
        exit(0)

    os.system(command)

    # Ask to copy into android assets
    copy = ask_copy_to_assets()
    if copy:
        print("Copying models to android app.")
        sourceDirectory = os.path.join(working_directory, "models-out")
        destinationDirectory = os.path.join(working_directory, "android", "app", "src", "main", "assets")
        shutil.copytree(sourceDirectory, destinationDirectory, dirs_exist_ok=True)
