#include <ecvl/augmentations.h>
#include <iostream>
#include <memory>
#include <string>

#include <ecvl/support_eddl.h>
#include <eddl/tensor/tensor.h>

#include "utils/utils.hpp"

int main(int argc, char **argv) {
  Arguments args = parse_arguments(argc, argv);

  // TODO: This is temporal, augs must be created following the argument
  // provided for them
  auto tr_augs = make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim({256, 256}, ecvl::InterpolationType::cubic));
  auto val_augs = make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim({256, 256}, ecvl::InterpolationType::cubic));
  auto te_augs = make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim({256, 256}, ecvl::InterpolationType::cubic));

  // Set the seed to reproduce the experiments
  ecvl::AugmentationParam::SetSeed(args.seed);
  ecvl::DatasetAugmentations dataset_augmentations{{tr_augs, val_augs, te_augs}};
  ecvl::DLDataset dataset(args.yaml_path, args.batch_size,
                          dataset_augmentations);
  std::cout << "Created ECVL DL Dataset from: \"" << args.yaml_path << "\"\n";

  // Get train split info
  dataset.SetSplit(ecvl::SplitType::training);
  int n_tr_samples = dataset.GetSplit().size();
  int n_tr_batches = n_tr_samples / args.batch_size;

  // Get validation split info
  dataset.SetSplit(ecvl::SplitType::validation);
  int n_val_samples = dataset.GetSplit().size();
  int n_val_batches = n_val_samples / args.batch_size;

  // Get test split info
  dataset.SetSplit(ecvl::SplitType::test);
  int n_te_samples = dataset.GetSplit().size();
  int n_te_batches = n_te_samples / args.batch_size;

  std::cout << "###################\n";
  std::cout << "# Dataset summary #\n";
  std::cout << "###################\n";
  std::cout << "Training split:\n";
  std::cout << " - n_samples = " << n_tr_samples << "\n";
  std::cout << " - n_batches = " << n_tr_batches
            << " (batch_size = " << args.batch_size << ")\n";
  std::cout << "Validation split:\n";
  std::cout << " - n_samples = " << n_val_samples << "\n";
  std::cout << " - n_batches = " << n_val_batches
            << " (batch_size = " << args.batch_size << ")\n";
  std::cout << "Testing split:\n";
  std::cout << " - n_samples = " << n_te_samples << "\n";
  std::cout << " - n_batches = " << n_te_batches
            << " (batch_size = " << args.batch_size << ")\n";

  return EXIT_SUCCESS;
}
