#include <iostream>
#include <memory>
#include <string>

#include <ecvl/augmentations.h>
#include <ecvl/support_eddl.h>
#include <eddl/tensor/tensor.h>

#include "training/augmentations.hpp"
#include "utils/utils.hpp"

int main(int argc, char **argv) {
  Arguments args = parse_arguments(argc, argv);

  // Set the seed to reproduce the experiments
  ecvl::AugmentationParam::SetSeed(args.seed);
  // Prepare data augmentations for each split
  ecvl::DatasetAugmentations dataset_augmentations{get_augmentations(args)};
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
  std::cout << "Input shape: {" << dataset.n_channels_ << ", "
            << args.target_shape[0] << ", " << args.target_shape[1] << "}\n";

  std::cout << "Batch size = " << args.batch_size << "\n";

  std::cout << "Classification labels:\n";
  for (int i = 0; i < dataset.classes_.size(); ++i)
    std::cout << " - " << i << ": " << dataset.classes_[i] << "\n";

  std::cout << "Training split:\n";
  std::cout << " - n_samples = " << n_tr_samples << "\n";
  std::cout << " - n_batches = " << n_tr_batches << "\n";
  std::cout << "Validation split:\n";
  std::cout << " - n_samples = " << n_val_samples << "\n";
  std::cout << " - n_batches = " << n_val_batches << "\n";
  std::cout << "Testing split:\n";
  std::cout << " - n_samples = " << n_te_samples << "\n";
  std::cout << " - n_batches = " << n_te_batches << "\n";

  return EXIT_SUCCESS;
}
