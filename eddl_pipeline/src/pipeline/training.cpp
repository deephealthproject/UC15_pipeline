#include "training.hpp"
#include <chrono>
#include <eddl/serialization/onnx/eddl_onnx.h>

void dataset_summary(ecvl::DLDataset &dataset, const Arguments &args) {
  // Get train split info
  dataset.SetSplit(ecvl::SplitType::training);
  const int n_tr_samples = dataset.GetSplit().size();
  const int n_tr_batches = n_tr_samples / args.batch_size;

  // Get validation split info
  dataset.SetSplit(ecvl::SplitType::validation);
  const int n_val_samples = dataset.GetSplit().size();
  const int n_val_batches = n_val_samples / args.batch_size;

  // Get test split info
  dataset.SetSplit(ecvl::SplitType::test);
  const int n_te_samples = dataset.GetSplit().size();
  const int n_te_batches = n_te_samples / args.batch_size;

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
}

TrainResults train(ecvl::DLDataset &dataset, Net *model,
                   const std::string &exp_name, Arguments &args) {
  // Get train split info
  dataset.SetSplit(ecvl::SplitType::training);
  const int n_tr_samples = dataset.GetSplit().size();
  const int n_tr_batches = n_tr_samples / args.batch_size;

  // Get validation split info
  dataset.SetSplit(ecvl::SplitType::validation);
  const int n_val_samples = dataset.GetSplit().size();
  const int n_val_batches = n_val_samples / args.batch_size;

  // Auxiliary variables to store the results
  vector<float> losses, accs, val_losses, val_accs;
  // To track the best models to store them in ONNX
  float best_loss = std::numeric_limits<float>::infinity();
  float best_acc = 0.f;
  // Paths to the current best checkpoints
  std::string best_model_byloss;
  std::string best_model_byacc;

  // Prepare experiment directory
  std::filesystem::path exp_path = args.exp_path;
  exp_path /= exp_name; // Append the exp_name to the experiments path
  std::filesystem::create_directories(exp_path);
  // Prepare the checkpoints folder inside the experiment folder
  const std::filesystem::path ckpts_path = exp_path / "ckpts";
  std::filesystem::create_directory(ckpts_path);

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
      const auto load_start = std::chrono::high_resolution_clock::now();
      dataset.LoadBatch(x, y);
      const auto load_end = std::chrono::high_resolution_clock::now();
      load_time += std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count();

      // Perform training
      const auto train_start = std::chrono::high_resolution_clock::now();
      eddl::train_batch(model, {x}, {y});
      const auto train_end = std::chrono::high_resolution_clock::now();
      train_time += std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start).count();

      // Get the current losses and metrics
      const float curr_loss = eddl::get_losses(model)[0];
      const float curr_acc = eddl::get_metrics(model)[0];
      losses.push_back(curr_loss);
      accs.push_back(curr_acc);

      // Show current stats
      std::cout << "\r";
      std::cout << "Batch " << b << "/" << n_tr_batches << ": ";
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Metrics[ loss=" << curr_loss << ", acc=" << curr_acc << " ]";
      std::cout << " - Timers[ ";
      std::cout << "avg_load_batch=" << (load_time / b) * 1e-6 << "s";
      std::cout << ", avg_train_batch=" << (train_time / b) * 1e-6 << "s ]";
      std::cout << std::flush;
    }
    const auto epoch_tr_end = std::chrono::high_resolution_clock::now();
    const float epoch_tr_time = std::chrono::duration_cast<std::chrono::microseconds>(epoch_tr_end - epoch_tr_start).count();
    std::cout << "\nEpoch " << e << " - training time elapsed = " << epoch_tr_time * 1e-6 << "s\n";

    // Reset the accumulated loss value
    eddl::reset_loss(model);

    // Validation phase
    load_time = 0.f;
    float eval_time = 0.f;
    const auto epoch_val_start = std::chrono::high_resolution_clock::now();
    std::cout << "\nEpoch " << e << " - validation:\n";
    dataset.SetSplit(ecvl::SplitType::validation);
    for (int b = 1; b <= n_val_batches; ++b) {
      // Load data
      const auto load_start = std::chrono::high_resolution_clock::now();
      dataset.LoadBatch(x, y);
      const auto load_end = std::chrono::high_resolution_clock::now();
      load_time += std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count();

      // Perform evaluation
      const auto eval_start = std::chrono::high_resolution_clock::now();
      eddl::eval_batch(model, {x}, {y});
      const auto eval_end = std::chrono::high_resolution_clock::now();
      eval_time += std::chrono::duration_cast<std::chrono::microseconds>(eval_end - eval_start).count();

      // Get the current losses and metrics
      const float curr_loss = eddl::get_losses(model)[0];
      const float curr_acc = eddl::get_metrics(model)[0];
      val_losses.push_back(curr_loss);
      val_accs.push_back(curr_acc);

      // Show current stats
      std::cout << "\r";
      std::cout << "Batch " << b << "/" << n_val_batches << ": ";
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Metrics[ val_loss=" << curr_loss << ", val_acc=" << curr_acc << " ]";
      std::cout << " - Timers[ ";
      std::cout << "avg_load_batch=" << (load_time / b) * 1e-6 << "s";
      std::cout << ", avg_eval_batch=" << (eval_time / b) * 1e-6 << "s ]";
      std::cout << std::flush;
    }
    const auto epoch_val_end = std::chrono::high_resolution_clock::now();
    const float epoch_val_time = std::chrono::duration_cast<std::chrono::microseconds>(epoch_val_end - epoch_val_start).count();
    std::cout << "\nEpoch " << e << " - validation time elapsed = " << epoch_val_time * 1e-6 << "s\n";

    // Get the final metrics from the validation split
    const float val_loss = val_losses.back();
    const float val_acc = val_accs.back();
    // Check if we have to save the current model as ONNX
    if (val_loss < best_loss || val_acc > best_acc) {
      // Prepare the onnx file name
      std::string onnx_name = exp_name;
      onnx_name += "_epoch-" + to_string(e);
      onnx_name += "_loss-" + to_string(val_loss);
      onnx_name += "_acc-" + to_string(val_acc);

      // Update the current best metrics and finish ONNX file name
      std::string onnx_fname;
      if (val_loss >= best_loss) { // Only improves acc
        best_acc = val_acc;

        onnx_name += "_by-acc.onnx";
        onnx_fname = (ckpts_path / onnx_name).string();
        best_model_byacc = onnx_fname;
        std::cout << "\nNew best model by acc: \"" << onnx_fname << "\"\n";
      } else if (val_acc <= best_acc) { // Only improves loss
        best_loss = val_loss;

        onnx_name += "_by-loss.onnx";
        onnx_fname = (ckpts_path / onnx_name).string();
        best_model_byloss = onnx_fname;
        std::cout << "\nNew best model by loss: \"" << onnx_fname << "\"\n";
      } else { // Improves loss and acc
        best_acc = val_acc;
        best_loss = val_loss;

        onnx_name += "_by-loss-and-acc.onnx";
        onnx_fname = (ckpts_path / onnx_name).string();
        best_model_byloss = onnx_fname;
        best_model_byacc = onnx_fname;
        std::cout << "\nNew best model by loss and acc: \"" << onnx_fname << "\"\n";
      }

      save_net_to_onnx_file(model, onnx_fname);
    }
  }

  // Free batch memory
  delete x;
  delete y;

  return TrainResults(losses, accs, val_losses, val_accs, best_model_byloss, best_model_byacc);
}

Optimizer *get_optimizer(const std::string &opt_name,
                         const float learning_rate) {
  if (opt_name == "Adam")
    return eddl::adam(learning_rate);
  if (opt_name == "SGD")
    return eddl::sgd(learning_rate, 0.9);

  std::cout << "The optimizer name provided (\"" << opt_name
            << "\") is not valid!\n";
  std::cout << "The valid optimizers are: Adam SGD\n";
  exit(EXIT_FAILURE);
}
