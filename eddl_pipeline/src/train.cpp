#include <iostream>
#include <memory>
#include <string>

#include <ecvl/augmentations.h>
#include <ecvl/support_eddl.h>
#include <eddl/apis/eddl.h>

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
  std::cout << "\nGoing to run the experiment: \"" << exp_name << "\"\n";

  // Set the seed to reproduce the experiments
  ecvl::AugmentationParam::SetSeed(args.seed);

  // Prepare data augmentations for each split
  ecvl::DatasetAugmentations data_augmentations{get_augmentations(args)};
  ecvl::DLDataset dataset(args.yaml_path, args.batch_size, data_augmentations);
  std::cout << "\nCreated ECVL DL Dataset from: \"" << args.yaml_path << "\"\n";

  // Create the model topology selected
  auto in_shape = {dataset.n_channels_, args.target_shape[0], args.target_shape[1]};
  Net *model = get_model(args.model, in_shape, dataset.classes_.size());

  Optimizer *opt = get_optimizer(args.optimizer, args.learning_rate);

  CompServ *cs;
  if (args.cpu)
    cs = eddl::CS_CPU(-1, "full_mem");
  else
    cs = eddl::CS_GPU(args.gpus, args.lsb, "full_mem");

  eddl::build(model, opt, {"softmax_cross_entropy"}, {"accuracy"}, cs);

  TrainResults tr_res = train(dataset, model, args);

  TestResults te_res = test(dataset, model, args);

  return EXIT_SUCCESS;
}
