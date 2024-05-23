# Installation

Clone the repository

```
git clone https://github.com/RivinHD/ImageInference.git
```

Then the submodules needed to be installed.

```
git submodule init
git submodule update --init --recursive
```

The `--init --recursive` is needed, because the submodule executorch also holds some submodules.

Then run the `install.sh` to install the needed requirements.\
**INFO:** Additional files will be added to the parent directory of the project.

**NOTE:** The `config.sh` holds the some path that you may need to set manually.
For now the only variable that can vary on system is the `QNN_SDK_ROOT`.
Which holds the path to the [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk).
For instruction on how to install see [Installing Qualcomm AI Engine Direct SDK](#installing-qualcomm-ai-engine-direct-sdk).

# Every time setup
To setup the project after the shell has been closed.
Use the `setup.sh` script.
Then you are ready to work with the project again.

# Installing Qualcomm AI Engine Direct SDK
1. Go to the website [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk).

2. Follow the download button. After logging in, search Qualcomm AI Stack at the Tool panel.

3. You can find Qualcomm AI Engine Direct SDK under the AI Stack group.

4. Please download the Linux version, and follow instructions on the page to extract the file.

5. The SDK should be installed to somewhere /opt/qcom/aistack/qnn by default.

If you place it somewhere else you needed to modify the path in the `config.sh`.
Also check if the version is the same i.e. the folder path `aistack/qairt/2.21.0.240401` exists otherwise you need to change the last folder to an available version folder.