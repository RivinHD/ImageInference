BasePath="$(dirname -- "${BASH_SOURCE[0]:-$0}")"  # relative
BasePath="$(cd -- "$BasePath" &> /dev/null && pwd 2> /dev/null)"  # absolutized and normalized
BasePath="$(dirname -- "$BasePath")" # go up one directory > ImageInference
BasePath="$(dirname -- "$BasePath")" # go up one directory > Parent of ImageInference

cd $BasePath/ImageInference/submodules/executorch

# Number of available processors
if [ "$(uname)" == "Darwin" ]; then
  CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
  CMAKE_JOBS=$(( $(nproc) - 1 ))
fi

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

SITE_PACKAGES="$(${PYTHON_EXECUTABLE} -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
CMAKE_INSTALL_PATH="${BasePath}/ImageInference/submodules/executorch/cmake-out"
CMAKE_PREFIX_PATH="${CMAKE_INSTALL_PATH}/lib/cmake/ExecuTorch;${SITE_PACKAGES}/torch/share/cmake/Torch;${SITE_PACKAGES}/torch/lib;"
BUILD_DIR="cmake-out/portable/custom_ops"

if [ "$1" == "clean" ]; then
  rm -rf "$BUILD_DIR"
  cmake "${BasePath}/ImageInference/backend/baremetal"\
      -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
      -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DCMAKE_BUILD_TYPE=Release \
      -B"$BUILD_DIR"
fi

cmake --build "$BUILD_DIR" -j "${CMAKE_JOBS}" --target install

if [ $? -eq 0 ]; then
  cd "$BUILD_DIR"
  ctest --output-on-failure -C Release #-j "${CMAKE_JOBS}"
fi

cd "$BasePath/ImageInference"