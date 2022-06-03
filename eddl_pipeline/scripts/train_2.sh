#!/bin/bash

if [ ! -d "build" ]; then
    echo "##################################"
    echo "## build directory not found!   ##"
    echo "## Going to compile the targets ##"
    echo "##################################"
    ./scripts/compile.sh
    echo ""
fi

# To find protobuf libraries
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"

echo "########################"
echo "## Executing training ##"
echo "########################"
#gdb --args ./build/bin/train_2 $@
./build/bin/train_2 $@
