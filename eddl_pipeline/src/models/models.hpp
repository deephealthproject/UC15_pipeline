#ifndef _MODELS_HPP_
#define _MODELS_HPP_

#include <eddl/apis/eddl.h>
#include <string>

Net *get_model(const std::string &model_name, const std::vector<int> &in_shape,
               const int num_classes);

#endif
