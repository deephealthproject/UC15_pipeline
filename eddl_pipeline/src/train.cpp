#include <iostream>
#include <memory>
#include <string>

#include <ecvl/augmentations.h>
#include <ecvl/support_eddl.h>
#include <eddl/apis/eddl.h>

#include "models/models.hpp"
#include "pipeline/augmentations.hpp"
#include "pipeline/training.hpp"
#include "pipeline/test.hpp"
#include "utils/utils.hpp"

int main(int argc, char **argv) {
  Arguments args = parse_arguments(argc, argv);

  // Set the seed to reproduce the experiments
  ecvl::AugmentationParam::SetSeed(args.seed);
  // Prepare data augmentations for each split
  ecvl::DatasetAugmentations data_augmentations{get_augmentations(args)};
  ecvl::DLDataset dataset(args.yaml_path, args.batch_size, data_augmentations);
  std::cout << "Created ECVL DL Dataset from: \"" << args.yaml_path << "\"\n";

  // Create the model topology selected
  auto in_shape = {dataset.n_channels_, args.target_shape[0],
                   args.target_shape[1]};
  Net *model = get_model(args.model, in_shape, dataset.classes_.size());

  // Compile the model
  eddl::build(model,
              eddl::adam(),
              {"softmax_cross_entropy"},
              {"accuracy"},
              eddl::CS_GPU({1}),
              true); // TODO Configurable opt, cs and initialization

  TrainResults tr_res = train(dataset, model, args);

  TestResults te_res = test(dataset, model, args);

  return EXIT_SUCCESS;
}
