#include "augmentations.hpp"
#include <iostream>

ecvl::DatasetAugmentations augmentations_v0_0(const Arguments &args) {
  // Training Split
  auto tr_augs = std::make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim(args.target_shape, ecvl::InterpolationType::cubic),
      ecvl::AugToFloat32(255.0));
  // Validation Split
  auto val_augs = std::make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim(args.target_shape, ecvl::InterpolationType::cubic),
      ecvl::AugToFloat32(255.0));
  // Testing Split
  auto te_augs = std::make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim(args.target_shape, ecvl::InterpolationType::cubic),
      ecvl::AugToFloat32(255.0));

  return ecvl::DatasetAugmentations({tr_augs, val_augs, te_augs});
}

ecvl::DatasetAugmentations augmentations_v1_0(const Arguments &args) {
  // Training Split
  auto tr_augs = std::make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim(args.target_shape, ecvl::InterpolationType::cubic),
      ecvl::AugMirror(0.5f),
      ecvl::AugRotate({-10, 10}),
      ecvl::AugBrightness({0, 50}),
      ecvl::AugGammaContrast({0.8, 1.2}),
      ecvl::AugToFloat32(255.0));
  // Validation Split
  auto val_augs = std::make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim(args.target_shape, ecvl::InterpolationType::cubic),
      ecvl::AugToFloat32(255.0));
  // Testing Split
  auto te_augs = std::make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim(args.target_shape, ecvl::InterpolationType::cubic),
      ecvl::AugToFloat32(255.0));

  return ecvl::DatasetAugmentations({tr_augs, val_augs, te_augs});
}

ecvl::DatasetAugmentations augmentations_v1_1(const Arguments &args) {
  // Training Split
  auto tr_augs = std::make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim(args.target_shape, ecvl::InterpolationType::cubic),
      ecvl::AugMirror(0.5f),
      ecvl::AugRotate({-15, 15}),
      ecvl::AugBrightness({0, 70}),
      ecvl::AugGammaContrast({0.6, 1.4}),
      ecvl::AugToFloat32(255.0));
  // Validation Split
  auto val_augs = std::make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim(args.target_shape, ecvl::InterpolationType::cubic),
      ecvl::AugToFloat32(255.0));
  // Testing Split
  auto te_augs = std::make_shared<ecvl::SequentialAugmentationContainer>(
      ecvl::AugResizeDim(args.target_shape, ecvl::InterpolationType::cubic),
      ecvl::AugToFloat32(255.0));

  return ecvl::DatasetAugmentations({tr_augs, val_augs, te_augs});
}

ecvl::DatasetAugmentations get_augmentations(const Arguments &args) {
  if (args.augmentations == "0.0")
    return augmentations_v0_0(args);
  if (args.augmentations == "1.0")
    return augmentations_v1_0(args);
  if (args.augmentations == "1.1")
    return augmentations_v1_1(args);

  std::cout << "The augmentations version provided (\"" << args.augmentations
            << "\") is not valid!\n";
  std::cout << "The valid versions are: 0.0, 1.0, 1.1\n";
  exit(EXIT_FAILURE);
}
