#include "training.hpp"
#include <chrono>

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
  Tensor *y = new Tensor({args.batch_size, static_cast<int>(dataset.classes_.size())});

  for (int e = 1; e <= args.epochs; ++e) {
    // Reset batch counter and shuffle all the data splits
    dataset.ResetAllBatches(true);

    // Reset the accumulated loss value
    eddl::reset_loss(model);

    // Training phase
    float load_time = 0.f;
    float train_time = 0.f;
    auto epoch_tr_start = std::chrono::high_resolution_clock::now();
    std::cout << "\nEpoch " << e << " - training:\n";
    dataset.SetSplit(ecvl::SplitType::training);
    for (int b = 1; b <= n_tr_batches; ++b) {
      // Load data
      auto load_start = std::chrono::high_resolution_clock::now();
      dataset.LoadBatch(x, y);
      auto load_end = std::chrono::high_resolution_clock::now();
      load_time += std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count();

      // Perform training
      auto train_start = std::chrono::high_resolution_clock::now();
      eddl::train_batch(model, {x}, {y});
      auto train_end = std::chrono::high_resolution_clock::now();
      train_time += std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start).count();

      // Get the current losses and metrics
      float curr_loss = eddl::get_losses(model)[0];
      float curr_acc = eddl::get_metrics(model)[0];

      // Show current stats
      std::cout << " Batch " << b << "/" << n_tr_batches << ": ";
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Metrics[ loss=" << curr_loss << ", acc=" << curr_acc << " ]";
      std::cout << " - Timers[ ";
      std::cout << "avg_load_batch=" << (load_time / b)  * 1e-6 << "s";
      std::cout << ", avg_train_batch=" << (train_time / b) * 1e-6 << "s ]";
      std::cout << std::endl;
    }
    auto epoch_tr_end = std::chrono::high_resolution_clock::now();
    float epoch_tr_time = std::chrono::duration_cast<std::chrono::microseconds>(epoch_tr_end - epoch_tr_start).count();
    std::cout << "Epoch " << e << " - training time elapsed = " << epoch_tr_time * 1e-6 << "s\n";

    // Reset the accumulated loss value
    eddl::reset_loss(model);

    // Validation phase
    load_time = 0.f;
    float eval_time = 0.f;
    auto epoch_val_start = std::chrono::high_resolution_clock::now();
    std::cout << "\nEpoch " << e << " - validation:\n";
    dataset.SetSplit(ecvl::SplitType::validation);
    for (int b = 1; b <= n_val_batches; ++b) {
      // Load data
      auto load_start = std::chrono::high_resolution_clock::now();
      dataset.LoadBatch(x, y);
      auto load_end = std::chrono::high_resolution_clock::now();
      load_time += std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count();

      // Perform evaluation
      auto eval_start = std::chrono::high_resolution_clock::now();
      eddl::eval_batch(model, {x}, {y});
      auto eval_end = std::chrono::high_resolution_clock::now();
      eval_time += std::chrono::duration_cast<std::chrono::microseconds>(eval_end - eval_start).count();

      // Get the current losses and metrics
      float curr_loss = eddl::get_losses(model)[0];
      float curr_acc = eddl::get_metrics(model)[0];

      // Show current stats
      std::cout << " Batch " << b << "/" << n_val_batches << ": ";
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Metrics[ loss=" << curr_loss << ", acc=" << curr_acc << " ]";
      std::cout << " - Timers[ ";
      std::cout << "avg_load_batch=" << (load_time / b)  * 1e-6 << "s";
      std::cout << ", avg_eval_batch=" << (eval_time / b) * 1e-6 << "s ]";
      std::cout << std::endl;
    }
    auto epoch_val_end = std::chrono::high_resolution_clock::now();
    float epoch_val_time = std::chrono::duration_cast<std::chrono::microseconds>(epoch_val_end - epoch_val_start).count();
    std::cout << "Epoch " << e << " - validation time elapsed = " << epoch_val_time * 1e-6 << "s\n";
  }

  // Free batch memory
  delete x;
  delete y;

  return TrainResults({}, {}, {}, {}, "", "");
}

Optimizer *get_optimizer(const std::string &opt_name, const float learning_rate) {
  if (opt_name == "Adam")
      return eddl::adam(learning_rate);
  if (opt_name == "SGD")
      return eddl::sgd(learning_rate, 0.9);

  std::cout << "The optimizer name provided (\"" << opt_name
            << "\") is not valid!\n";
  std::cout << "The valid optimizers are: Adam SGD\n";
  exit(EXIT_FAILURE);
}
