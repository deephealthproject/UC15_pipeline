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

Net *model_2a(const std::vector<int> in_shape, const int num_classes)
{
    Layer *in_ = eddl::Input(in_shape);
    Layer *l_ = in_; // Auxiliar pointer

    float l2_alpha = 1.0e-4;

    // 40 ch block with reduction
    l_ = eddl::Conv2D(l_, 40, {7, 7}, {2, 2}, "valid");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    l_ = eddl::Conv2D(l_, 60, {3, 3}, {1, 1}, "same");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    // 60 ch block with reduction
    l_ = eddl::Conv2D(l_, 60, {5, 5}, {2, 2}, "valid");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    // 120 ch block
    l_ = eddl::Conv2D(l_, 80, {3, 3}, {1, 1}, "same");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    l_ = eddl::Conv2D(l_, 80, {3, 3}, {1, 1}, "same");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);

    for (int i = 0; i < 3; i++) {
        // 240 ch block with reduction
        l_ = eddl::Conv2D(l_, 140, {5, 5}, {2, 2}, "valid");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        // 240 ch block 
        l_ = eddl::Conv2D(l_, 140, {3, 3}, {1, 1}, "same");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        l_ = eddl::Conv2D(l_, 140, {3, 3}, {1, 1}, "same");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
    }
    for (int i = 0; i < 2; i++) {
        // 480 ch block with reduction
        l_ = eddl::Conv2D(l_, 280, {5, 5}, {2, 2}, "valid");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        // 480 ch block 
        l_ = eddl::Conv2D(l_, 280, {3, 3}, {1, 1}, "same");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        l_ = eddl::Conv2D(l_, 280, {3, 3}, {1, 1}, "same");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
    }
    l_ = eddl::Conv2D(l_, 512, {3, 3}, {1, 1}, "valid");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    l_ = eddl::Conv2D(l_, 1024, {3, 3}, {1, 1}, "valid");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);

    // Dense block
    l_ = eddl::Flatten(l_);
    l_ = eddl::ReLu(eddl::L2(eddl::Dense(l_, 1024), l2_alpha));

    // Output layer
    //Layer *out_ = eddl::Softmax(eddl::Dense(l_, num_classes));
    Layer *out_ = eddl::Sigmoid(eddl::Dense(l_, num_classes));

    return eddl::Model({in_}, {out_});
}

Net *model_2b(const std::vector<int> in_shape, const int num_classes)
{
    Layer *in_ = eddl::Input(in_shape);
    Layer *l_ = in_; // Auxiliar pointer

    float l2_alpha = 1.0e-4;

    // 40 ch block with reduction
    l_ = eddl::Conv2D(l_, 32, {7, 7}, {1, 1}, "same");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    l_ = eddl::MaxPool2D(l_, {2, 2}, {2, 2});

    // 120 ch block
    for (int i = 0; i < 3; i++) {
        l_ = eddl::Conv2D(l_, 64, {3, 3}, {1, 1}, "same");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        l_ = eddl::Conv2D(l_, 64, {3, 3}, {1, 1}, "same");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        l_ = eddl::MaxPool2D(l_, {2, 2}, {2, 2});
    }

    for (int i = 0; i < 3; i++) {
        // 240 ch block 
        l_ = eddl::Conv2D(l_, 128, {3, 3}, {1, 1}, "same");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        l_ = eddl::Conv2D(l_, 128, {3, 3}, {1, 1}, "same");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        l_ = eddl::MaxPool2D(l_, {2, 2}, {2, 2});
    }
    for (int i = 0; i < 2; i++) {
        // 480 ch block 
        l_ = eddl::Conv2D(l_, 256, {3, 3}, {1, 1}, "same");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        l_ = eddl::Conv2D(l_, 256, {3, 3}, {1, 1}, "same");
        l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        l_ = eddl::MaxPool2D(l_, {2, 2}, {2, 2});
    }

    // Dense block
    l_ = eddl::Flatten(l_);
    l_ = eddl::ReLu(eddl::L2(eddl::Dense(l_, 512), l2_alpha));
    l_ = eddl::ReLu(eddl::L2(eddl::Dense(l_, 512), l2_alpha));


    // Output layer
    //Layer *out_ = eddl::Softmax(eddl::Dense(l_, num_classes));
    Layer *out_ = eddl::Sigmoid(eddl::Dense(l_, num_classes));

    return eddl::Model({in_}, {out_});
}


Net *get_model(const std::string &model_name, const std::vector<int> &in_shape,
               const int num_classes) {

  if (model_name == "model_1")  return model_1(in_shape, num_classes);
  if (model_name == "model_2a") return model_2a(in_shape, num_classes);
  if (model_name == "model_2b") return model_2b(in_shape, num_classes);

  std::cout << "The model name provided (\"" << model_name
            << "\") is not valid!\n";
  std::cout << "The valid names are: model_1\n";
  exit(EXIT_FAILURE);
}
