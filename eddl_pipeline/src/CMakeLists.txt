# ##############################################################################
# Dependencies
# ##############################################################################
find_package(EDDL REQUIRED)
find_package(ecvl REQUIRED)
# add_subdirectory(third_party)
include_directories(third_party/cxxopts/include)

# ##############################################################################
# Aditional sources to include
# ##############################################################################
# add_subdirectory(models)
add_subdirectory(utils)

# ##############################################################################
# Executables
# ##############################################################################
# Store the executables in the build/bin folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(TRAIN_EXECUTABLE "train")
add_executable(${TRAIN_EXECUTABLE} train.cpp)
target_link_libraries(${TRAIN_EXECUTABLE} EDDL::eddl ECVL_CORE ECVL_EDDL
                      DATASET utils)
