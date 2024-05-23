# Set the base path
BasePath="$(dirname "${BASH_SOURCE[0]}")"  # relative
BasePath="$(cd "$BasePath" && pwd)"  # absolutized and normalized
BasePath="$(dirname "$BasePath")" # go up one directory

# Export the required environment variables
export QNN_SDK_ROOT="/opt/qcom/aistack/qairt/2.21.0.240401"
export ANDROID_NDK="${BasePath}/android/ndk/android-ndk-r26d"
export EXECUTORCH_ROOT="submodules/executorch"
export FLATC_EXECUTABLE="${EXECUTORCH_ROOT}/third-party/flatbuffers/cmake-out/flatc"
export LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang/":$LD_LIBRARY_PATH

echo "QNN_SDK_ROOT: $QNN_SDK_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"
echo "EXECUTORCH_ROOT: $EXECUTORCH_ROOT"
echo "FLATC_EXECUTABLE: $FLATC_EXECUTABLE"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"