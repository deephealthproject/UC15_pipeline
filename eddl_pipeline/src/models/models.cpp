#include "models.hpp"
#include <iostream>

Net *model_1(const std::vector<int> in_shape, const int num_classes, const std::string & classifier_output) {
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

Layer * block_2a(Layer * l_in, int num_filters, float l2_alpha)
{
    Layer * l = l_in;

    l = eddl::Dropout(l, 0.5f);
    l = eddl::Conv2D(l, num_filters, {5, 5}, {2, 2}, "valid");
    l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);

    return l;
}

Net *model_2a(const std::vector<int> in_shape, const int num_classes, const std::string & classifier_output)
{
    Layer *in_ = eddl::Input(in_shape);
    Layer *l_ = in_; // Auxiliar pointer

    float l2_alpha = 1.0e-5;

    int num_filters = 37;
    l_ = eddl::Conv2D(l_, num_filters, {13, 13}, {2, 2}, "valid");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);

    num_filters <<= 1;
    l_ = block_2a(l_, num_filters, l2_alpha);
    num_filters <<= 1;
    l_ = block_2a(l_, num_filters, l2_alpha);
    l_ = block_2a(l_, num_filters, l2_alpha);
    l_ = block_2a(l_, num_filters, l2_alpha);
    num_filters <<= 1;
    l_ = block_2a(l_, num_filters, l2_alpha);
    l_ = block_2a(l_, num_filters, l2_alpha);

    num_filters <<= 1;
    l_ = eddl::Conv2D(l_, num_filters, {3, 3}, {1, 1}, "valid");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    /*
    l_ = eddl::Conv2D(l_, num_filters, {2, 2}, {1, 1}, "valid");
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    */

    // Dense block
    l_ = eddl::Flatten(l_);
    l_ = eddl::Dropout(l_, 0.5f);
    l_ = eddl::Dense(l_, 1024);
    l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    l_ = eddl::Dropout(l_, 0.5f);

    // Output layer
    Layer *out_ = nullptr;
    if (classifier_output == "sigmoid")
        out_ = eddl::Sigmoid(eddl::Dense(l_, num_classes == 2 ? 1 : num_classes));
    else
        out_ = eddl::Softmax(eddl::Dense(l_, num_classes));

    return eddl::Model({in_}, {out_});
}

Net *model_2b(const std::vector<int> in_shape, const int num_classes, const std::string & classifier_output)
{
    Layer *in_ = eddl::Input(in_shape);
    Layer *l_ = in_; // Auxiliar pointer

    float l2_alpha = 1.0e-6;

    //l_ = eddl::Conv2D(l_, 64, {13, 13}, {2, 2}, "valid");
    l_ = eddl::Conv2D(l_, 64, {13, 13}, {1, 1}, "valid");
    //l_ = eddl::Conv2D(l_, 32, {3, 3}, {1, 1}, "same");
    //l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    l_ = eddl::MaxPool2D(l_, {2, 2}, {2, 2});
    //l_ = eddl::AvgPool2D(l_, {2, 2}, {2, 2});

    l_ = eddl::Conv2D(l_, 128, {3, 3}, {1, 1}, "same");
    //l_ = eddl::L2(l_, l2_alpha);
    l_ = eddl::ReLu(l_);
    l_ = eddl::MaxPool2D(l_, {2, 2}, {2, 2});
    //l_ = eddl::AvgPool2D(l_, {2, 2}, {2, 2});

    int num_filters = 128;
    //   1 ->   2 ->   3 ->   4 ->    5
    //  64 -> 128 -> 256 -> 512 -> 1024 -> 2048
    // 512 -> 256 -> 128 ->  64 ->  32  ->   16
    for (int i = 1; i <= 6; i++) {
        int nf = min(num_filters, 1024);
        l_ = eddl::Conv2D(l_, nf, {3, 3}, {1, 1}, "same");
        //l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        l_ = eddl::Conv2D(l_, nf, {1, 1}, {1, 1}, "same");
        //l_ = eddl::L2(l_, l2_alpha);
        l_ = eddl::ReLu(l_);
        l_ = eddl::MaxPool2D(l_, {2, 2}, {2, 2});
        //l_ = eddl::AvgPool2D(l_, {2, 2}, {2, 2});
        num_filters += 128;
    }

    // Dense block
    l_ = eddl::Flatten(l_);
    l_ = eddl::ReLu(eddl::L2(eddl::Dense(l_, 1024), l2_alpha));
    l_ = eddl::ReLu(eddl::L2(eddl::Dense(l_, 512), l2_alpha));


    // Output layer
    Layer *out_ = eddl::Softmax(eddl::Dense(l_, num_classes));

    return eddl::Model({in_}, {out_});
}
Net *model_2c(const std::vector<int> in_shape, const int num_classes, const std::string & classifier_output)
{
    Layer *in_ = eddl::Input(in_shape);
    Layer *l_ = in_; // Auxiliar pointer

    l_ = eddl::Flatten(l_);
    l_ = eddl::ReLu(eddl::Dense(l_, 1024));
    l_ = eddl::ReLu(eddl::Dense(l_, 1024));


    // Output layer
    Layer *out_ = eddl::Softmax(eddl::Dense(l_, num_classes));

    return eddl::Model({in_}, {out_});
}
Layer * block_3a(Layer * l_in, int num_filters, float l2_alpha)
{
    Layer * l = l_in;
    l = eddl::Dropout(l, 0.5);
    l = eddl::Conv2D(l, num_filters, {3, 3}, {1, 1}, "same");
    l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    l = eddl::MaxPool2D(l, {2, 2}, {2, 2});
    return l;
}
Net *model_3a(const std::vector<int> in_shape, const int num_classes, const std::string & classifier_output)
{
    Layer *in_ = eddl::Input(in_shape);

    float l2_alpha = 1.0e-6;

    Layer * i_a = eddl::Conv2D(in_, 16, { 3,  3}, {1, 1}, "same"); 
    Layer * i_b = eddl::Conv2D(in_, 16, { 7,  7}, {1, 1}, "same");
    Layer * i_c = eddl::Conv2D(in_, 16, {11, 11}, {1, 1}, "same");

    // 256 x 256

    Layer * l_1_ab = eddl::Concat({i_a, i_b});
    Layer * l_1_bc = eddl::Concat({i_b, i_c});
    Layer * l_1_ca = eddl::Concat({i_c, i_a});

    l_1_ab = block_3a(l_1_ab, 32, l2_alpha);
    l_1_bc = block_3a(l_1_bc, 32, l2_alpha);
    l_1_ca = block_3a(l_1_ca, 32, l2_alpha);

    // 128 x 128

    Layer * l_2_a = eddl::Concat({l_1_ca, l_1_ab, block_3a(i_a, 32, l2_alpha)});
    Layer * l_2_b = eddl::Concat({l_1_ab, l_1_bc, block_3a(i_b, 32, l2_alpha)});
    Layer * l_2_c = eddl::Concat({l_1_bc, l_1_ca, block_3a(i_c, 32, l2_alpha)});

    l_2_a = block_3a(l_2_a, 64, l2_alpha);
    l_2_b = block_3a(l_2_b, 64, l2_alpha);
    l_2_c = block_3a(l_2_c, 64, l2_alpha);

    // 64 x 64

    Layer * l_3_a = eddl::Concat({l_2_b, l_2_c});
    Layer * l_3_b = eddl::Concat({l_2_a, l_2_c});
    Layer * l_3_c = eddl::Concat({l_2_a, l_2_b});

    l_3_a = block_3a(l_3_a, 64, l2_alpha);
    l_3_b = block_3a(l_3_b, 64, l2_alpha);
    l_3_c = block_3a(l_3_c, 64, l2_alpha);

    // 32 x 32

    Layer * l_4_a = eddl::Concat({l_3_b, l_3_c});
    Layer * l_4_b = eddl::Concat({l_3_a, l_3_c});
    Layer * l_4_c = eddl::Concat({l_3_a, l_3_b});

    l_4_a = block_3a(l_4_a, 64, l2_alpha);
    l_4_b = block_3a(l_4_b, 64, l2_alpha);
    l_4_c = block_3a(l_4_c, 64, l2_alpha);

    // 16 x 16

    Layer * l_5_a = eddl::Concat({l_4_b, l_4_c});
    Layer * l_5_b = eddl::Concat({l_4_a, l_4_c});
    Layer * l_5_c = eddl::Concat({l_4_a, l_4_b});

    l_5_a = block_3a(l_5_a, 64, l2_alpha);
    l_5_b = block_3a(l_5_b, 64, l2_alpha);
    l_5_c = block_3a(l_5_c, 64, l2_alpha);

    // 8 x 8

    Layer * l_6_a = eddl::Concat({l_5_b, l_5_c});
    Layer * l_6_b = eddl::Concat({l_5_a, l_5_c});
    Layer * l_6_c = eddl::Concat({l_5_a, l_5_b});

    l_6_a = block_3a(l_6_a, 128, l2_alpha);
    l_6_b = block_3a(l_6_b, 128, l2_alpha);
    l_6_c = block_3a(l_6_c, 128, l2_alpha);

    // 4 x 4

    Layer * l_7_a = eddl::Concat({l_6_b, l_6_c});
    Layer * l_7_b = eddl::Concat({l_6_a, l_6_c});
    Layer * l_7_c = eddl::Concat({l_6_a, l_6_b});

    l_7_a = block_3a(l_7_a, 128, l2_alpha);
    l_7_b = block_3a(l_7_b, 128, l2_alpha);
    l_7_c = block_3a(l_7_c, 128, l2_alpha);

    Layer * l = eddl::Concat({l_7_a, l_7_b, l_7_c});

    l = block_3a(l, 128, l2_alpha);

    // Dense block
    l = eddl::Flatten(l);
    l = eddl::Dropout(l, 0.5);
    //l = eddl::ReLu(eddl::L2(eddl::Dense(l, 1024), l2_alpha));
    l = eddl::ReLu(eddl::L2(eddl::Dense(l,  512), l2_alpha));
    l = eddl::Dropout(l, 0.5);


    // Output layer
    // Output layer
    Layer *out_ = nullptr;
    if (classifier_output == "sigmoid")
        out_ = eddl::Sigmoid(eddl::Dense(l, num_classes == 2 ? 1 : num_classes));
    else
        out_ = eddl::Softmax(eddl::Dense(l, num_classes));

    return eddl::Model({in_}, {out_});
}
Layer * block_3b(Layer * l_in, int num_filters, float l2_alpha, float dropout_rate)
{
    Layer * l = l_in;

    if (dropout_rate > 0.0)
        l = eddl::Dropout(l, dropout_rate);
    l = eddl::Conv2D(l, num_filters, {3, 3}, {1, 1}, "same");
    l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    l = eddl::MaxPool2D(l, {2, 2}, {2, 2});
    return l;
}
Net *model_3b(const std::vector<int> in_shape, const int num_classes, const std::string & classifier_output)
{
    Layer *in_ = eddl::Input(in_shape);

    float l2_alpha = 1.0e-6f;
    float dropout_rate = 0.5f;

    Layer * i_a = eddl::Conv2D(in_, 16, { 3,  3}, {1, 1}, "same"); 
    Layer * i_b = eddl::Conv2D(in_, 32, { 7,  7}, {1, 1}, "same");
    Layer * i_c = eddl::Conv2D(in_, 32, {11, 11}, {1, 1}, "same");

    Layer * l = eddl::Concat({i_a, i_b, i_c}); // 512 x 512

    l = block_3b(l,  64, l2_alpha, dropout_rate); // 256 x 256
    l = block_3b(l,  64, l2_alpha, dropout_rate); // 128 x 128
    l = block_3b(l, 128, l2_alpha, dropout_rate); // 64 x 64
    l = block_3b(l, 128, l2_alpha, dropout_rate); // 32 x 32
    l = block_3b(l, 256, l2_alpha, dropout_rate); // 16 x 16
    l = block_3b(l, 256, l2_alpha, dropout_rate); // 8 x 8
    l = block_3b(l, 512, l2_alpha, dropout_rate); // 4 x 4
    l = block_3b(l, 512, l2_alpha, dropout_rate); // 2 x 2

    // Dense block
    l = eddl::Flatten(l);
    if (dropout_rate > 0.0) l = eddl::Dropout(l, dropout_rate);
    l = eddl::ReLu(eddl::L2(eddl::Dense(l, 1024), l2_alpha));
    if (dropout_rate > 0.0) l = eddl::Dropout(l, dropout_rate);

    // Output layer
    Layer *out_ = nullptr;
    if (classifier_output == "sigmoid")
        out_ = eddl::Sigmoid(eddl::Dense(l, num_classes == 2 ? 1 : num_classes));
    else
        out_ = eddl::Softmax(eddl::Dense(l, num_classes));

    return eddl::Model({in_}, {out_});
}

Layer * descending_block_qu_ex(Layer * l_in, int num_filters, float l2_alpha, float dropout_rate)
{
    Layer * l = l_in;

    if (dropout_rate > 0.0) l = eddl::Dropout(l, dropout_rate);
    l = eddl::Conv2D(l, num_filters, {3, 3}, {1, 1}, "same");
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    l = eddl::MaxPool2D(l, {2, 2}, {2, 2});
    return l;
}
Layer * ascending_block_qu_ex(Layer * l_in, int num_filters, float l2_alpha, float dropout_rate)
{
    Layer * l = l_in;

    if (dropout_rate > 0.0) l = eddl::Dropout(l, dropout_rate);
    l = eddl::UpSampling2D(l, {2, 2});
    l = eddl::Conv2D(l, num_filters, {3, 3}, {1, 1}, "same");
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    return l;
}

Net * model_covid_qu_ex_01(const std::vector<int> in_shape, const int num_classes)
{
    float l2_alpha = 0.0f; // 1.0e-5f;
    float dropout_rate = 0.0f; // 0.5f;

    Layer * in_ = eddl::Input(in_shape);

    Layer * l_down = in_;
    l_down = descending_block_qu_ex(l_down,  64, l2_alpha, 0.0f        ); // 256 x 256
    l_down = descending_block_qu_ex(l_down,  64, l2_alpha, dropout_rate); // 128 x 128
    l_down = descending_block_qu_ex(l_down, 128, l2_alpha, dropout_rate); //  64 x  64
    l_down = descending_block_qu_ex(l_down, 128, l2_alpha, dropout_rate); //  32 x  32
    l_down = descending_block_qu_ex(l_down, 256, l2_alpha, dropout_rate); //  16 x  16
    l_down = descending_block_qu_ex(l_down, 256, l2_alpha, dropout_rate); //   8 x   8
    //l_down = descending_block_qu_ex(l_down, 256, l2_alpha, dropout_rate); //   4 x   4

    Layer * l = l_down;
    l = eddl::Conv2D(l, 256, {3, 3}, {1, 1}, "same");
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    l = eddl::Conv2D(l, 256, {3, 3}, {1, 1}, "same");
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);

    Layer * bottleneck = l;

    Layer * l_up = l;
    //l_up = ascending_block_qu_ex(l_up, 256, l2_alpha, dropout_rate); //   4 x   4
    l_up = ascending_block_qu_ex(l_up, 256, l2_alpha, dropout_rate); //   8 x   8
    l_up = ascending_block_qu_ex(l_up, 256, l2_alpha, dropout_rate); //  16 x  16
    l_up = ascending_block_qu_ex(l_up, 128, l2_alpha, dropout_rate); //  32 x  32
    l_up = ascending_block_qu_ex(l_up, 128, l2_alpha, dropout_rate); //  64 x  64
    l_up = ascending_block_qu_ex(l_up,  64, l2_alpha, dropout_rate); // 128 x 128
    l_up = ascending_block_qu_ex(l_up,  64, l2_alpha, dropout_rate); // 256 x 256


    Layer * mask_out_ = eddl::Sigmoid(eddl::Conv2D(l_up, 1, {1, 1}, {1, 1}, "same"));
    //Layer * mask_out_ = eddl::Conv2D(l_up, 1, {1, 1}, {1, 1}, "same");

    l = eddl::Flatten(bottleneck);
    if (dropout_rate > 0.0) l = eddl::Dropout(l, dropout_rate);
    l = eddl::Dense(l, 2048);
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    if (dropout_rate > 0.0) l = eddl::Dropout(l, dropout_rate);
    l = eddl::Dense(l, 1024);
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    if (dropout_rate > 0.0) l = eddl::Dropout(l, dropout_rate);
    l = eddl::Dense(l, 512);
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    l = eddl::Dense(l, num_classes);
    Layer * clf_out_ = eddl::Softmax(l);

    return eddl::Model({in_}, {mask_out_, clf_out_});
}
Net * model_covid_qu_ex_02(const std::vector<int> in_shape, const int num_classes)
{
    float l2_alpha = 0.0f; // 1.0e-5f;
    float dropout_rate = 0.0f; // 0.5f;

    Layer * in_ = eddl::Input(in_shape);

    int filters = 32;

    Layer * l_down_256 = in_; // 256 x 256
    Layer * l_down_128 = descending_block_qu_ex(l_down_256, filters * 1, l2_alpha, 0.0f        ); // 128 x 128
    Layer * l_down_064 = descending_block_qu_ex(l_down_128, filters * 2, l2_alpha, dropout_rate); //  64 x  64
    Layer * l_down_032 = descending_block_qu_ex(l_down_064, filters * 2, l2_alpha, dropout_rate); //  32 x  32
    Layer * l_down_016 = descending_block_qu_ex(l_down_032, filters * 4, l2_alpha, dropout_rate); //  16 x  16
    Layer * l_down_008 = descending_block_qu_ex(l_down_016, filters * 4, l2_alpha, dropout_rate); //   8 x   8
    Layer * l_down_004 = descending_block_qu_ex(l_down_008, filters * 8, l2_alpha, dropout_rate); //   4 x   4

    Layer * l = l_down_004;
    l = eddl::Conv2D(l, filters * 4, {3, 3}, {1, 1}, "same");
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    l = eddl::Conv2D(l, filters * 4, {3, 3}, {1, 1}, "same");
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);

    Layer * bottleneck = l;

    Layer * l_up_004 = bottleneck;
    //l_up_004 = eddl::Concat({l_up_004, l_down_004});
    Layer * l_up_008 = ascending_block_qu_ex(l_up_004, filters * 8, l2_alpha, dropout_rate); //   8 x   8
    l_up_008 = eddl::Concat({l_up_008, l_down_008});
    Layer * l_up_016 = ascending_block_qu_ex(l_up_008, filters * 4, l2_alpha, dropout_rate); //  16 x  16
    l_up_016 = eddl::Concat({l_up_016, l_down_016});
    Layer * l_up_032 = ascending_block_qu_ex(l_up_016, filters * 4, l2_alpha, dropout_rate); //  32 x  32
    l_up_032 = eddl::Concat({l_up_032, l_down_032});
    Layer * l_up_064 = ascending_block_qu_ex(l_up_032, filters * 2, l2_alpha, dropout_rate); //  64 x  64
    l_up_064 = eddl::Concat({l_up_064, l_down_064});
    Layer * l_up_128 = ascending_block_qu_ex(l_up_064, filters * 2, l2_alpha, dropout_rate); // 128 x 128
    l_up_128 = eddl::Concat({l_up_128, l_down_128});
    Layer * l_up_256 = ascending_block_qu_ex(l_up_128, filters * 1, l2_alpha, dropout_rate); // 256 x 256


    Layer * mask_out_ = eddl::Sigmoid(eddl::Conv2D(l_up_256, 1, {1, 1}, {1, 1}, "same"));
    //Layer * mask_out_ = eddl::Conv2D(l_up_256, 1, {1, 1}, {1, 1}, "same");

    l = eddl::Flatten(bottleneck);
    if (dropout_rate > 0.0) l = eddl::Dropout(l, dropout_rate);
    l = eddl::Dense(l, 1024);
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    if (dropout_rate > 0.0) l = eddl::Dropout(l, dropout_rate);
    l = eddl::Dense(l, 1024);
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    if (dropout_rate > 0.0) l = eddl::Dropout(l, dropout_rate);
    l = eddl::Dense(l, 512);
    if (l2_alpha > 0.0) l = eddl::L2(l, l2_alpha);
    l = eddl::ReLu(l);
    l = eddl::Dense(l, num_classes);
    Layer * clf_out_ = eddl::Softmax(l);

    return eddl::Model({in_}, {mask_out_, clf_out_});
}

std::tuple<Net *, bool, std::vector<std::string>>
resnet(const std::vector<int> in_shape,
       const int num_classes,
       const int version,
       const std::string &classifier_output,
       const bool pretrained) {
  eddl::model pretrained_model;
  switch (version) {
      case 18:
        pretrained_model = eddl::download_resnet18(true, in_shape);
        break;
      case 34:
        pretrained_model = eddl::download_resnet34(true, in_shape);
        break;
      case 50:
        pretrained_model = eddl::download_resnet50(true, in_shape);
        break;
      case 101:
        pretrained_model = eddl::download_resnet101(true, in_shape);
        break;
      case 152:
        pretrained_model = eddl::download_resnet152(true, in_shape);
        break;
      default:
        std::cerr << "[ERROR] ResNet version " << version << " is not valid!\n";
        break;
  }

  // Get the input layer of the pretrained model
  eddl::layer in_ = eddl::getLayer(pretrained_model, "input");
  // Get the last layer of the pretrained model
  eddl::layer top_layer = eddl::getLayer(pretrained_model, "top");

  // Create the new densely connected part
  std::vector<std::string> layers2init{"dense1", "dense_out"};
  const int input_units = top_layer->output->shape[1];
  eddl::layer l = eddl::Dense(top_layer, input_units / 2, true, layers2init[0]);
  l = eddl::ReLu(l);
  l = eddl::Dropout(l, 0.4);
  // Output layer
  eddl::layer out_ = nullptr;
  if (classifier_output == "sigmoid")
    out_ = eddl::Sigmoid(eddl::Dense(l, num_classes == 2 ? 1 : num_classes,
                                     true, layers2init[1]));
  else
    out_ = eddl::Softmax(eddl::Dense(l, num_classes, true, layers2init[1]));

  return {eddl::Model({in_}, {out_}), !pretrained, layers2init};
}

std::tuple<Net *, bool, std::vector<std::string>>
get_model(const std::string &model_name,
          const std::vector<int> &in_shape,
          const int num_classes,
          const std::string & classifier_output) {

  // Models from scratch
  if (model_name == "model_1")
    return {model_1( in_shape, num_classes, classifier_output), true, {}};
  if (model_name == "model_2a")
    return {model_2a(in_shape, num_classes, classifier_output), true, {}};
  if (model_name == "model_2b")
    return {model_2b(in_shape, num_classes, classifier_output), true, {}};
  if (model_name == "model_2c")
    return {model_2c(in_shape, num_classes, classifier_output), true, {}};
  if (model_name == "model_3a")
    return {model_3a(in_shape, num_classes, classifier_output), true, {}};
  if (model_name == "model_3b")
    return {model_3b(in_shape, num_classes, classifier_output), true, {}};

  // ResNet models (not pretrained)
  if (model_name == "ResNet18")
    return resnet(in_shape, num_classes, 18, classifier_output, false);
  if (model_name == "ResNet34")
    return resnet(in_shape, num_classes, 34, classifier_output, false);
  if (model_name == "ResNet50")
    return resnet(in_shape, num_classes, 50, classifier_output, false);
  if (model_name == "ResNet101")
    return resnet(in_shape, num_classes, 101, classifier_output, false);
  if (model_name == "ResNet152")
    return resnet(in_shape, num_classes, 152, classifier_output, false);

  // ResNet models (pretrained with Imagenet)
  if (model_name == "Pretrained_ResNet18")
    return resnet(in_shape, num_classes, 18, classifier_output, true);
  if (model_name == "Pretrained_ResNet34")
    return resnet(in_shape, num_classes, 34, classifier_output, true);
  if (model_name == "Pretrained_ResNet50")
    return resnet(in_shape, num_classes, 50, classifier_output, true);
  if (model_name == "Pretrained_ResNet101")
    return resnet(in_shape, num_classes, 101, classifier_output, true);
  if (model_name == "Pretrained_ResNet152")
    return resnet(in_shape, num_classes, 152, classifier_output, true);

  if (model_name == "covid19-qu-ex-01") return {model_covid_qu_ex_01(in_shape, num_classes), true, {}};
  if (model_name == "covid19-qu-ex-02") return {model_covid_qu_ex_02(in_shape, num_classes), true, {}};

  std::cout << "The model name provided (\"" << model_name
            << "\") is not valid!\n";
  std::cout << "The valid names are: model_1\n";
  exit(EXIT_FAILURE);
}
