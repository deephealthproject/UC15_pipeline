#include "training.hpp"
#include <chrono>
#include <eddl/serialization/onnx/eddl_onnx.h>
#include <fstream>

#include "data_generator.hpp"

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

void apply_regularization(Net *model, const std::string &regularization, const float factor) {
  if (regularization == "l1")
    for (eddl::layer l : model->layers)
      eddl::L1(l, factor);
  else if (regularization == "l2")
    for (eddl::layer l : model->layers)
      eddl::L2(l, factor);
  else if (regularization == "l1l2")
    for (eddl::layer l : model->layers)
      eddl::L1L2(l, factor, factor);
}

TrainResults train(ecvl::DLDataset &dataset, Net *model,
                   const std::string &exp_name, Arguments &args) {
  // Prepare the train data generator
  dataset.SetSplit(ecvl::SplitType::training);
  const int n_tr_samples = dataset.GetSplit().size();
  const int n_tr_batches = n_tr_samples / args.batch_size;

  // Prepare the validation data generator
  dataset.SetSplit(ecvl::SplitType::validation);
  const int n_val_batches = dataset.GetNumBatches(ecvl::SplitType::validation);

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

  // Store the full experiment configuration in a json file
  std::ofstream json_args((exp_path / "args.json").string());
  json_args << args;
  json_args.close();

  bool avoid_unfreeze = false; // To avoid unfreezing the weights more than once

  for (int e = 1; e <= args.epochs; ++e) {
    std::cout << "Starting epoch " << e << "/" << args.epochs << ":\n";

    if (!avoid_unfreeze) {
      // Check if the model must be unfreezed
      if (args.is_pretrained && args.frozen_epochs < e && args.frozen_epochs > 0) {
        std::cout << "Going to unfreeze the pretrained weights\n";
        for (const std::string &lname : args.pretrained_layers)
          eddl::setTrainable(model, lname, true);
        avoid_unfreeze = true;
      }
    }

    float new_lr = args.learning_rate * (1.0 / (1.0 + args.lr_decay * (e - 1)));
    eddl::setlr(model, {new_lr});
    std::cout << std::scientific;
    std::cout << "Current learning rate: " << new_lr << "\n";

    dataset.SetSplit(ecvl::SplitType::training);
    dataset.ResetAllBatches(true); // Reset batch counter and shuffle all the data splits
    dataset.Start();

    // Reset the accumulated loss value
    eddl::reset_loss(model);

    int counter_samples = 0;
    float sum_class_0 = 0;

    // Training phase
    float load_time = 0.f;
    float train_time = 0.f;
    float curr_loss = -1.f;
    float curr_acc = -1.f;
    auto epoch_tr_start = std::chrono::high_resolution_clock::now();
    for (int b = 1; b <= n_tr_batches; ++b) {
      // Load data
      const auto load_start = std::chrono::high_resolution_clock::now();
      auto [samples, x, y] = dataset.GetBatch();
      for (int i = 0; i < y->shape[0]; ++i, ++counter_samples) {
        sum_class_0 += y->ptr[i * y->stride[0]];
      }
      const auto load_end = std::chrono::high_resolution_clock::now();
      load_time += std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count();

      // Perform training
      const auto train_start = std::chrono::high_resolution_clock::now();
      if (args.classifier_output == "sigmoid" && dataset.classes_.size() == 2) {
          Tensor *aux_y = y.get()->select({":", "1"});
          aux_y->reshape_({-1});
          eddl::train_batch(model, {x.get()}, {aux_y});
          delete aux_y;
      } else {
          eddl::train_batch(model, {x.get()}, {y.get()});
      }
      const auto train_end = std::chrono::high_resolution_clock::now();
      train_time += std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start).count();

      // Get the current losses and metrics
      curr_loss = eddl::get_losses(model)[0];
      curr_acc = eddl::get_metrics(model)[0];

      // Show current stats
      std::cout << "Training: Epoch " << e << "/" << args.epochs << " - ";
      std::cout << "Batch " << b << "/" << n_tr_batches << ": ";
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Metrics[ loss=" << curr_loss << ", acc=" << curr_acc << " ]";
      std::cout << " - Timers[ ";
      std::cout << "avg_load_batch=" << (load_time / b) * 1e-6 << "s";
      std::cout << ", avg_train_batch=" << (train_time / b) * 1e-6 << "s ]";
      std::cout << " - DataGenerator[ |fifo| = " << dataset.GetQueueSize() << " ]";
      std::cout << std::endl;
    }
    dataset.Stop();
    const auto epoch_tr_end = std::chrono::high_resolution_clock::now();
    const float epoch_tr_time = std::chrono::duration_cast<std::chrono::microseconds>(epoch_tr_end - epoch_tr_start).count();
    std::cout << "Epoch " << e << "/" << args.epochs << ": training time elapsed = " << epoch_tr_time * 1e-6 << "s\n";
    std::cout << "Result of blind assignment to class 0: " << sum_class_0 << " / " << counter_samples << " = " << sum_class_0 / counter_samples << "\n\n";

    // Store the train split metrics for the current epoch
    losses.push_back(curr_loss);
    accs.push_back(curr_acc);

    // Reset the accumulated loss value
    eddl::reset_loss(model);

    counter_samples = 0;
    sum_class_0 = 0;

    // Validation phase
    load_time = 0.f;
    float eval_time = 0.f;
    curr_loss = -1.f;
    curr_acc = -1.f;
    const auto epoch_val_start = std::chrono::high_resolution_clock::now();
    dataset.SetSplit(ecvl::SplitType::validation);
    dataset.Start();
    for (int b = 1; b <= n_val_batches; ++b) {
      // Load data
      const auto load_start = std::chrono::high_resolution_clock::now();
      auto [samples, x, y] = dataset.GetBatch();
      for (int i = 0; i < y->shape[0]; ++i, ++counter_samples) {
        sum_class_0 += y->ptr[i * y->stride[0]];
      }
      const auto load_end = std::chrono::high_resolution_clock::now();
      load_time += std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count();

      // We make sure that the shape of the model is the correct one
      // This may change for the last batch
      model->resize(x.get()->shape[0]);

      // Perform evaluation
      const auto eval_start = std::chrono::high_resolution_clock::now();
      if (args.classifier_output == "sigmoid" && dataset.classes_.size() == 2) {
          Tensor *aux_y = y.get()->select({":", "1"});
          aux_y->reshape_({-1});
          eddl::eval_batch(model, {x.get()}, {aux_y});
          delete aux_y;
      } else {
          eddl::eval_batch(model, {x.get()}, {y.get()});
      }
      const auto eval_end = std::chrono::high_resolution_clock::now();
      eval_time += std::chrono::duration_cast<std::chrono::microseconds>(eval_end - eval_start).count();

      // Get the current losses and metrics
      curr_loss = eddl::get_losses(model)[0];
      curr_acc = eddl::get_metrics(model)[0];

      // Show current stats
      std::cout << "Validation: Epoch " << e << "/" << args.epochs << " - ";
      std::cout << "Batch " << b << "/" << n_val_batches << ": ";
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Metrics[ val_loss=" << curr_loss << ", val_acc=" << curr_acc << " ]";
      std::cout << " - Timers[ ";
      std::cout << "avg_load_batch=" << (load_time / b) * 1e-6 << "s";
      std::cout << ", avg_eval_batch=" << (eval_time / b) * 1e-6 << "s ]";
      std::cout << " - DataGenerator[ |fifo| = " << dataset.GetQueueSize() << " ]";
      std::cout << std::endl;
    }
    dataset.Stop();
    const auto epoch_val_end = std::chrono::high_resolution_clock::now();
    const float epoch_val_time = std::chrono::duration_cast<std::chrono::microseconds>(epoch_val_end - epoch_val_start).count();
    std::cout << "Epoch " << e << "/" << args.epochs << ": validation time elapsed = " << epoch_val_time * 1e-6 << "s\n";
    std::cout << "Result of blind assignment to class 0: " << sum_class_0 << " / " << counter_samples << " = " << sum_class_0 / counter_samples << "\n\n";

    // Store the validation split metrics for the current epoch
    val_losses.push_back(curr_loss);
    val_accs.push_back(curr_acc);

    // Check if we have to save the current model as ONNX
    if (curr_loss < best_loss || curr_acc > best_acc) {
      // Prepare the onnx file name
      std::string onnx_name = exp_name;
      onnx_name += "_epoch-" + to_string(e);
      onnx_name += "_loss-" + to_string(curr_loss);
      onnx_name += "_acc-" + to_string(curr_acc);

      // Update the current best metrics and finish ONNX file name
      std::string onnx_fname;
      if (curr_loss >= best_loss) { // Only improves acc
        best_acc = curr_acc;

        onnx_name += "_by-acc.onnx";
        onnx_fname = (ckpts_path / onnx_name).string();
        best_model_byacc = onnx_fname;
        std::cout << "New best model by acc: \"" << onnx_fname << "\"\n\n";
      } else if (curr_acc <= best_acc) { // Only improves loss
        best_loss = curr_loss;

        onnx_name += "_by-loss.onnx";
        onnx_fname = (ckpts_path / onnx_name).string();
        best_model_byloss = onnx_fname;
        std::cout << "New best model by loss: \"" << onnx_fname << "\"\n\n";
      } else { // Improves loss and acc
        best_acc = curr_acc;
        best_loss = curr_loss;

        onnx_name += "_by-loss-and-acc.onnx";
        onnx_fname = (ckpts_path / onnx_name).string();
        best_model_byloss = onnx_fname;
        best_model_byacc = onnx_fname;
        std::cout << "New best model by loss and acc: \"" << onnx_fname << "\"\n\n";
      }

      save_net_to_onnx_file(model, onnx_fname);
    }

    // Show the epoch results for each split
    std::cout << "Results: Epoch " << e << "/" << args.epochs << ": ";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Training[ loss=" << losses.back() << ", acc=" << accs.back() << " ] - ";
    std::cout << "Validation[ val_loss=" << val_losses.back() << ", val_acc=" << val_accs.back() << " ]\n\n";

  auto results = TrainResults(losses, accs, val_losses, val_accs, best_model_byloss, best_model_byacc);
  // Store the training history in a CSV
  results.save_hist_to_csv((exp_path / "train_res.csv").string());
  }

  auto results = TrainResults(losses, accs, val_losses, val_accs, best_model_byloss, best_model_byacc);

  return results;
}

Optimizer *get_optimizer(const std::string &opt_name,
                         const float learning_rate) {
  if (opt_name == "Adam")    return eddl::adam(learning_rate);
  if (opt_name == "SGD")     return eddl::sgd(learning_rate, 0.9);
  if (opt_name == "RMSprop") return eddl::rmsprop(learning_rate);

  std::cout << "The optimizer name provided (\"" << opt_name
            << "\") is not valid!\n";
  std::cout << "The valid optimizers are: Adam SGD\n";
  exit(EXIT_FAILURE);
}

void TrainResults::save_hist_to_csv(const std::string &csv_path) const {
  std::ofstream out_csv(csv_path);
  // Set the CSV header
  out_csv << "epoch,loss,acc,val_loss,val_acc\n";
  // Add a row for each epoch
  for (int e = 0; e < losses.size(); ++e) {
    out_csv << e + 1 << ",";
    out_csv << losses[e] << "," << accs[e] << ",";  // Train split
    out_csv << val_losses[e] << "," << val_accs[e]; // Validation split
    if (e != losses.size()) out_csv << "\n";
  }
  out_csv.close();
}
