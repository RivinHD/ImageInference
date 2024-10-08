# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

import os
import shutil

MODELS = [
    "resnet50v1.5",
]

HTP_QUALCOMM = "Qualcomm (HTP)"
CUSTOM = "Custom"

BACKEND = [
    "XNNPACK (CPU)",
    # "VULKAN (GPU)",
    HTP_QUALCOMM,
    CUSTOM,
]

QUALCOMM_DEVICES = [
    "SM8650 (Snapdragon 8 Gen 3)",
    "SM8550 (Snapdragon 8 Gen 2)",
    "SM8475 (Snapdragon 8+ Gen 1)",
    "SM8450 (Snapdragon 8 Gen 1)",
]

QUANTIZATION = [
    "false",
    "int8",
    "int16",
    "int16int4",
    "int8int4",
]

MODELS_TO_PYFILE = {
    "resnet50v1.5": "resnet50v15",
}

MODELS_TO_AOT_LIB = {
    "resnet50v1.5": "libbaremetal_ops_aot_lib.so",
}

BACKEND_TO_DIRECTORY = {
    "XNNPACK (CPU)": "execu_python.xnnpack",
    "VULKAN (GPU)": "execu_python.vulkan",
    HTP_QUALCOMM: "execu_python.htp",
    CUSTOM: "baremetal",
}

EXECU_PACKAGE = "backend"


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


def select_backend() -> list[str]:
    backend = []
    while len(backend) == 0:
        print("Select Backend to run on separated by comma or 'all' to support every backend.")
        print(f"Available backend are {BACKEND}")
        text = input("> ").strip()
        if text == 'all':
            backend = BACKEND
        else:
            for h in text.split(","):
                h = h.strip().lower()
                for hw in BACKEND:
                    if hw.lower().startswith(h):
                        backend.append(hw)
    print(f"Selected backend are {backend}")
    return backend


def ask_correct_backend() -> bool:
    text = input("Is the selected backend correct? (y/n) > ").strip()
    while not (text.startswith("y") or text.startswith("n")):
        text = input("Please enter y or n > ")
    return text.startswith("y")


def select_htp_devices() -> str:
    devices = []
    while len(devices) == 0:
        print("Since you want to use HTP you need to select Qualcomm device to run on.")
        print("Select devices to run on separated by comma or 'all' to support every listed devices.")
        print(f"Available devices are {QUALCOMM_DEVICES}")
        text = input("> ").strip()
        if text == 'all':
            devices = [qd.split(" ")[0] for qd in QUALCOMM_DEVICES]
        else:
            input_devices = text.split(",")
            for device in input_devices:
                for qd in QUALCOMM_DEVICES:
                    qdr = qd.split(" ")[0].lower()
                    if device.strip().lower() == qdr:
                        devices.append(qd.split(" ")[0])
                        break
    print(f"Selected device is {devices}")
    return devices


def ask_correct_devices() -> bool:
    text = input("Is the selected devices correct? (y/n) > ").strip()
    while not (text.startswith("y") or text.startswith("n")):
        text = input("Please enter y or n > ")
    return text.startswith("y")


def ask_quantization() -> tuple[bool, bool]:
    quantization = []
    while len(quantization) == 0:
        print("Select quantization to apply separated by comma or 'all' to support every listed quantization.")
        print("If multiple types are listed the first one is for the activation functions and the second one is"
              + "for the weights. E.g. int16int4 uses int16 for the activation functions and int4 for the weights.")
        print(f"Available quantization is {QUANTIZATION}")
        text = input("> ").strip()
        if text == 'all':
            quantization = QUANTIZATION
        else:
            input_quant = text.split(",")
            for quant in input_quant:
                for qt in QUANTIZATION:
                    if quant.strip().lower() == qt:
                        quantization.append(qt)
                        break
    print(f"Selected quantization is {quantization}")
    return quantization


def ask_correct_quantization() -> bool:
    text = input("Is the selected quantization correct? (y/n) > ").strip()
    while not (text.startswith("y") or text.startswith("n")):
        text = input("Please enter y or n > ")
    return text.startswith("y")


def ask_start_processing() -> bool:
    text = input("Start processing? (y/n) > ").strip().lower()
    while not (text.startswith("y") or text.startswith("n")):
        text = input("Please enter y or n > ")
    return text.startswith("y")


def generate_command(
        models: list[str],
        hardware: list[str],
        quantize: list[str],
        htp_devices: list[str] = []) -> str:
    executorch_path = os.getenv('EXECUTORCH_ROOT')
    lib_path = os.path.join(executorch_path, "cmake-out", "lib")
    commands = []
    for model in models:
        for quant in quantize:
            for hw in hardware:
                package_path = ".".join([EXECU_PACKAGE, BACKEND_TO_DIRECTORY[hw], MODELS_TO_PYFILE[model]])
                base_command = f"python -m {package_path} --quantize {quant}"
                if hw == HTP_QUALCOMM:
                    for device in htp_devices:
                        commands.append(f"{base_command} --model {device}")
                elif hw == CUSTOM:
                    lib_file = os.path.join(lib_path, MODELS_TO_AOT_LIB[model])
                    commands.append(f"{base_command} --so_library {lib_file}")
                else:
                    commands.append(base_command)
    return "; ".join(commands)


def ask_copy_to_assets() -> bool:
    text = input("Should the generated models by copied to the android app? (y/n) > ").strip().lower()
    while not (text.startswith("y") or text.startswith("n")):
        text = input("Please enter y or n > ")
    return text.startswith("y")


if __name__ == "__main__":

    if "EXECUTORCH_ROOT" not in os.environ:
        raise RuntimeError("Environment variable EXECUTORCH_ROOT must be set")
    print(f"EXECUTORCH_ROOT={os.getenv('EXECUTORCH_ROOT')}")

    # Ask for models
    models = select_models()
    models_correct = ask_correct_model()
    while not models_correct:
        models = select_models()
        models_correct = ask_correct_model()
    print()

    # Ask for hardware
    backend = select_backend()
    backend_correct = ask_correct_backend()
    while not backend_correct:
        backend = select_backend()
        backend_correct = ask_correct_backend()
    print()

    htp_devices = ""
    if HTP_QUALCOMM in backend:
        htp_devices = select_htp_devices()
        htp_devices_correct = ask_correct_devices()
        while not htp_devices_correct:
            backend = select_htp_devices()
            backend_correct = ask_correct_devices()
        print()

    # Ask for quantization
    quantization = ask_quantization()
    quantization_correct = ask_correct_quantization()
    while not quantization_correct:
        quantization = ask_quantization()
        quantization_correct = ask_correct_quantization()
    print()

    # Generate command and start processing
    command = generate_command(models, backend, quantization, htp_devices)
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
