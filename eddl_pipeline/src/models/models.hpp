#ifndef _MODELS_HPP_
#define _MODELS_HPP_

#include <eddl/apis/eddl.h>
#include <string>
#include <tuple>

std::tuple<Net *, bool, std::vector<std::string>>
get_model(const std::string &model_name, const std::vector<int> &in_shape,
          const int num_classes, const std::string &classifier_output);

#endif
