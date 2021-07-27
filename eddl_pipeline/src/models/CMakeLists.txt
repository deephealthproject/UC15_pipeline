add_library(models STATIC models.hpp models.cpp)
target_link_libraries(models EDDL::eddl)
