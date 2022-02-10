add_library(utils STATIC utils.hpp utils.cpp)
target_link_libraries(utils ECVL_EDDL EDDL::eddl)
