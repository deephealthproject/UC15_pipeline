#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi
pushd build > /dev/null # Enter the build directory to compile

echo ""
echo "###################"
echo "## Running CMake ##"
echo "###################"
#cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX $@ ..
CONDA_PREFIX="/usr/local"
cmake	-DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
	-DCMAKE_CUDA_ARCHITECTURES="all" \
	-DCMAKE_CUDA_COMPILER="/usr/local/cuda-10.2/bin/nvcc" \
	$@ ..

echo ""
echo "#########################"
echo "## Compiling (make -j) ##"
echo "#########################"
make VERBOSE=1 -j

popd > /dev/null # Go back to the root directory
