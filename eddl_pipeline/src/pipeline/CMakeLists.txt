add_library(augmentations STATIC augmentations.hpp augmentations.cpp)
add_library(training STATIC training.hpp training.cpp)
target_link_libraries(augmentations ECVL_CORE ECVL_EDDL)
target_link_libraries(training EDDL::eddl ECVL_CORE ECVL_EDDL)
