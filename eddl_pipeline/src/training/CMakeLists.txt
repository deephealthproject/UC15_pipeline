add_library(augmentations STATIC augmentations.hpp augmentations.cpp)
target_link_libraries(augmentations ECVL_CORE ECVL_EDDL)
