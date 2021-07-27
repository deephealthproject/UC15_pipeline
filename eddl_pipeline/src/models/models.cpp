#include "models.hpp"
#include <iostream>

Net *model_1(const std::vector<int> in_shape, const int num_classes) {
  Layer *in_ = eddl::Input(in_shape);
  Layer *l_ = in_; // Auxiliar pointer

  // 32 ch block
  l_ = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(l_, 32, {3, 3})));
  l_ = eddl::MaxPool2D(l_, {2, 2});
  // 64 ch block
  l_ = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(l_, 64, {3, 3})));
  l_ = eddl::MaxPool2D(l_, {2, 2});
  l_ = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(l_, 64, {3, 3})));
  l_ = eddl::MaxPool2D(l_, {2, 2});
  // 128 ch block
  l_ = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(l_, 128, {3, 3})));
  l_ = eddl::MaxPool2D(l_, {2, 2});
  l_ = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(l_, 128, {3, 3})));
  l_ = eddl::MaxPool2D(l_, {2, 2});
  // 256 ch block
  l_ = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(l_, 256, {3, 3})));
  l_ = eddl::MaxPool2D(l_, {2, 2});
  // Dense block
  l_ = eddl::Flatten(l_);
  l_ = eddl::ReLu(eddl::Dense(l_, 512));
  Layer *out_ = eddl::Softmax(eddl::Dense(l_, num_classes));

  return eddl::Model({in_}, {out_});
}

Net *get_model(const std::string &model_name, const std::vector<int> &in_shape,
               const int num_classes) {
  if (model_name == "model_1")
    return model_1(in_shape, num_classes);

  std::cout << "The model name provided (\"" << model_name
            << "\") is not valid!\n";
  std::cout << "The valid names are: model_1\n";
  exit(EXIT_FAILURE);
}
