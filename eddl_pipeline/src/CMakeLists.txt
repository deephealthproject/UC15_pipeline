# ##############################################################################
# Dependencies
# ##############################################################################
find_package(EDDL REQUIRED)
find_package(ecvl REQUIRED)
find_package(OpenMP)
#include_directories(${cxxopts_SOURCE_DIR}/include) # To handle the arguments
include_directories(third_party/cxxopts/include) # To handle the arguments


# ##############################################################################
# Aditional sources to include
# ##############################################################################
add_subdirectory(utils)
add_subdirectory(pipeline)
add_subdirectory(models)
add_subdirectory(third_party/cxxopts)

# ##############################################################################
# Executables
# ##############################################################################
# Store the executables in the build/bin folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(TRAIN_EXECUTABLE "train")
add_executable(${TRAIN_EXECUTABLE} train.cpp)
target_link_libraries(
  ${TRAIN_EXECUTABLE}
  EDDL::eddl
  ECVL_CORE
  ECVL_EDDL
  DATASET
  utils
  augmentations
  training
  test
  models)


set(TRAIN_DISTR_EXECUTABLE "train_2")
add_executable(${TRAIN_DISTR_EXECUTABLE} train_2.cpp)
target_link_libraries(
  ${TRAIN_DISTR_EXECUTABLE}
  EDDL::eddl
  ECVL_CORE
  ECVL_EDDL
  DATASET
  utils
  augmentations
  training
  test
  models)

set(TRAIN_DISTR_EXECUTABLE "train_distr")
add_executable(${TRAIN_DISTR_EXECUTABLE} train_distr.cpp)
target_link_libraries(
  ${TRAIN_DISTR_EXECUTABLE}
  EDDL::eddl
  ECVL_CORE
  ECVL_EDDL
  DATASET
  utils
  augmentations
  training
  test
  models)
