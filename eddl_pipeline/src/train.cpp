#include <iostream>
#include <memory>
#include <string>

#include <ecvl/augmentations.h>
#include <ecvl/support_eddl.h>
#include <eddl/apis/eddl.h>
#include <eddl/serialization/onnx/eddl_onnx.h>

#include "models/models.hpp"
#include "pipeline/augmentations.hpp"
#include "pipeline/test.hpp"
#include "pipeline/training.hpp"
#include "utils/utils.hpp"

int main(int argc, char **argv) {
  Arguments args = parse_arguments(argc, argv);

  // Set the experiment name for logs
  std::string exp_name = get_current_time_str();
  exp_name += "_net-" + args.model;
  exp_name += "_DA-" + args.augmentations;
  exp_name += "_input-" + to_string(args.target_shape[0]) + "x" + to_string(args.target_shape[1]);
  exp_name += "_opt-" + args.optimizer;
  exp_name += "_lr-" + to_string(args.learning_rate);
  std::cout << "Going to run the experiment \"" << exp_name << "\"\n";

  // Set the seed to reproduce the experiments
  ecvl::AugmentationParam::SetSeed(args.seed);

  // Prepare data augmentations for each split
  ecvl::DatasetAugmentations data_augmentations{get_augmentations(args)};
  ecvl::ColorType color_type = get_color_type(args.rgb_or_gray);

  ecvl::DLDataset dataset(args.yaml_path, args.batch_size, data_augmentations,
                          color_type, ecvl::ColorType::GRAY, args.workers);
  std::cout << "\nCreated ECVL DL Dataset from \"" << args.yaml_path << "\"\n\n";
  // Show basic dataset info
  dataset_summary(dataset, args);

  std::cout << "\n";
  std::cout << "##################\n";
  std::cout << "# Model creation #\n";
  std::cout << "##################\n";
  Net *model;
  bool init_weights = true;
  if (args.ckpt.empty()) {
    // Create the model topology selected
    std::cout << "Going to prepare the model \"" << args.model << "\"\n";
    auto in_shape = {dataset.n_channels_, args.target_shape[0], args.target_shape[1]};
    auto [net, init_w, l2init] = get_model(args.model, in_shape, dataset.classes_.size(),
                                           args.classifier_output);
    model = net;
    init_weights = init_w;
    args.layers2init = l2init;
    std::cout << "Model created!\n";
  } else {
    // Load the ONNX model as checkpoint
    std::cout << "Going to load the model from \"" << args.ckpt << "\"\n";
    model = import_net_from_onnx_file(args.ckpt);
    init_weights = false; // Avoid to override the imported weights
    std::cout << "Model loaded!\n";
  }

  if (!args.regularization.empty()) {
    std::cout << "Going to apply " << args.regularization << " regularization to the model\n";
    apply_regularization(model, args.regularization, args.regularization_factor);
  }

  Optimizer *opt = get_optimizer(args.optimizer, args.learning_rate);

  CompServ *cs = get_computing_service(args);

  if (args.classifier_output == "softmax") {
    eddl::build(model, opt, {"softmax_cross_entropy"}, {"accuracy"}, cs, init_weights);
  } else if (args.classifier_output == "sigmoid") {
    if (dataset.classes_.size() == 2) {
      eddl::build(model, opt, {"binary_cross_entropy"}, {"binary_accuracy"}, cs, init_weights);
    } else {
      eddl::build(model, opt, {"mse"}, {"categorical_accuracy"}, cs, init_weights);
      //eddl::build(model, opt, {"cross_entropy"}, {"categorical_accuracy"}, cs, init_weights);
    }
  } else {
    std::cerr << "ERROR: unexpected classifier output: " << args.classifier_output << std::endl;
    std::abort();
  }
  std::cout << "Model built!\n\n";

  // Check if we have to prepare a pretrained model
  if (!init_weights && args.ckpt.empty()) {
    args.is_pretrained = true;
    // Freeze the pretrained layers
    if (args.frozen_epochs > 0) {
      std::cout << "Going to freeze the pretrained weights\n";
      for (eddl::layer l : model->layers) {
        if (std::find(args.layers2init.begin(), args.layers2init.end(), l->name) == args.layers2init.end())
          eddl::setTrainable(model, l->name, false);
          args.pretrained_layers.push_back(l->name);
      }
    }

    // Initialize the new layers
    for (const std::string& lname : args.layers2init)
      eddl::initializeLayer(model, lname);
  }

  eddl::summary(model);

  std::cout << "###############\n";
  std::cout << "# Train phase #\n";
  std::cout << "###############\n";
  TrainResults tr_res = train(dataset, model, exp_name, args);
  delete model; // Free the memory before the test phase

  std::cout << "##############\n";
  std::cout << "# Test phase #\n";
  std::cout << "##############\n";
  std::vector<std::string> test_models_paths = {tr_res.best_model_by_loss};
  if (tr_res.best_model_by_loss != tr_res.best_model_by_acc)
    test_models_paths.push_back(tr_res.best_model_by_acc);

  for (auto &onnx_path : test_models_paths) {
    std::cout << "Going to run test with model \"" << onnx_path << "\"\n\n";
    // Load the model for testing
    model = import_net_from_onnx_file(onnx_path);

    // Build the model
    opt = get_optimizer(args.optimizer, args.learning_rate);
    cs = get_computing_service(args);

    if (args.classifier_output == "softmax") {
      eddl::build(model, opt, {"softmax_cross_entropy"}, {"accuracy"}, cs, false);
    } else if (args.classifier_output == "sigmoid") {
      if (dataset.classes_.size() == 2) {
        eddl::build(model, opt, {"binary_cross_entropy"}, {"binary_accuracy"}, cs, false);
      } else {
        eddl::build(model, opt, {"mse"}, {"categorical_accuracy"}, cs, false);
        //eddl::build(model, opt, {"cross_entropy"}, {"categorical_accuracy"}, cs, false);
      }
    } else {
      std::cerr << "ERROR: unexpected classifier output: " << args.classifier_output << std::endl;
      std::abort();
    }

    // We create the dataset again to avoid an issue that stops the testing process
    // This issue affects the testing phase of the second model tested (by acc)
    ecvl::DLDataset dataset(args.yaml_path, args.batch_size, data_augmentations,
                            color_type, ecvl::ColorType::GRAY, args.workers);

    TestResults te_res = test(dataset, model, args);

    delete model; // Free memory before the next test iteration
  }

  return EXIT_SUCCESS;
}
