#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi
pushd build > /dev/null # Enter the build directory to compile

echo ""
echo "###################"
echo "## Running CMake ##"
echo "###################"
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX $@ ..

echo ""
echo "#########################"
echo "## Compiling (make -j) ##"
echo "#########################"
make -j

popd > /dev/null # Go back to the root directory
