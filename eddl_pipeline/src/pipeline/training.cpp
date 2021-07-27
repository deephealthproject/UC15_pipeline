#include "training.hpp"

TrainResults train(ecvl::DLDataset &dataset, Net *model, Arguments &args) {
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

  // Auxiliary variables to store the results
  // TODO

  // Auxiliary tensors to load the batches
  Tensor *x = new Tensor({args.batch_size, dataset.n_channels_,
                          args.target_shape[0], args.target_shape[1]});
  Tensor *y =
      new Tensor({args.batch_size, static_cast<int>(dataset.classes_.size())});

  for (int e = 0; e < args.epochs; ++e) {
    // Reset batch counter and shuffle all the data splits
    dataset.ResetAllBatches(true);

    // Reset the accumulated loss value
    eddl::reset_loss(model);

    std::cout << "Epoch " << e << ":\n";
    // Training phase
    dataset.SetSplit(ecvl::SplitType::training);
    for (int b = 0; b < n_tr_batches; ++b) {
      dataset.LoadBatch(x, y);

      eddl::train_batch(model, {x}, {y});

      std::cout << "Batch ";
      eddl::print_loss(model, b); // Show current loss and metrics
      std::cout << std::endl;
    }

    // Reset the accumulated loss value
    eddl::reset_loss(model);

    // Validation phase
    std::cout << "Validation:\n";
    dataset.SetSplit(ecvl::SplitType::validation);
    for (int b = 0; b < n_val_batches; ++b) {
      dataset.LoadBatch(x, y);

      eddl::eval_batch(model, {x}, {y});

      std::cout << "Batch ";
      eddl::print_loss(model, b); // Show current loss and metrics
      std::cout << std::endl;
    }
  }

  // Free batch memory
  delete x;
  delete y;

  return TrainResults({}, {}, {}, {}, "", "");
}
