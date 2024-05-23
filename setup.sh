# Set the base path
BasePath="$(dirname -- "${BASH_SOURCE[0]}")"  # relative
BasePath="$(cd -- "$BasePath" && pwd)"  # absolutized and normalized
BasePath="$(dirname -- "$BasePath")" # go up one directory
echo "Setup the project with BasePath at $BasePath"

# Run the setup
cd ImageInference
git submodule init
git submodule sync
git submodule update --init --recursive

# Enable conda environment
eval "$(conda shell.bash hook)"
conda activate imageinfernce

# Setup the configuartion
./config.sh