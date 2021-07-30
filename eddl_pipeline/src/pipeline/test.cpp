#include "test.hpp"
#include <chrono>

#include "data_generator.hpp"

TestResults test(ecvl::DLDataset &dataset, Net *model, Arguments &args) {
  if (args.use_dldataset)
    return test_dataset(dataset, model, args);
  else
    return test_datagen(dataset, model, args);
}

TestResults test_dataset(ecvl::DLDataset &dataset, Net *model, Arguments &args) {
  // Get test split info
  dataset.SetSplit(ecvl::SplitType::test);
  const int n_te_samples = dataset.GetSplit().size();
  const int n_te_batches = n_te_samples / args.batch_size;

  // Auxiliary tensors to load the batches
  Tensor *x = new Tensor({args.batch_size, dataset.n_channels_,
                          args.target_shape[0], args.target_shape[1]});
  Tensor *y = new Tensor({args.batch_size, static_cast<int>(dataset.classes_.size())});

  // Reset batch counter and shuffle all the data splits
  dataset.ResetAllBatches();

  // Reset the accumulated loss value
  eddl::reset_loss(model);

  // Auxiliary variables to store the results
  float loss, acc;

  // Validation phase
  float load_time = 0.f;
  float eval_time = 0.f;
  const auto test_start = std::chrono::high_resolution_clock::now();
  dataset.SetSplit(ecvl::SplitType::test);
  for (int b = 1; b <= n_te_batches; ++b) {
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

    // Show current stats
    std::cout << "Testing: Batch " << b << "/" << n_te_batches << ": ";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Metrics[ loss=" << curr_loss << ", acc=" << curr_acc << " ]";
    std::cout << " - Timers[ ";
    std::cout << "avg_load_batch=" << (load_time / b)  * 1e-6 << "s";
    std::cout << ", avg_eval_batch=" << (eval_time / b) * 1e-6 << "s ]";
    std::cout << std::endl;
  }
  const auto test_end = std::chrono::high_resolution_clock::now();
  const float test_time = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_start).count();
  std::cout << "Test time elapsed = " << test_time * 1e-6 << "s\n\n";

  // Free batch memory
  delete x;
  delete y;

  return TestResults(loss, acc);
}

TestResults test_datagen(ecvl::DLDataset &dataset, Net *model, Arguments &args) {
  // Get test split info
  dataset.SetSplit(ecvl::SplitType::test);
  const int n_te_samples = dataset.GetSplit().size();
  const int n_te_batches = n_te_samples / args.batch_size;
  DataGenerator datagen(&dataset, args.batch_size, args.target_shape,
                        {static_cast<int>(dataset.classes_.size())},
                        args.workers);

  // Reset batch counter
  dataset.ResetAllBatches();

  // Reset the accumulated loss value
  eddl::reset_loss(model);

  // Auxiliary variables to store the results
  float loss, acc;

  // Validation phase
  float load_time = 0.f;
  float eval_time = 0.f;
  const auto test_start = std::chrono::high_resolution_clock::now();
  datagen.Start();
  for (int b = 1; datagen.HasNext(); ++b) {
    // Load data
    const auto load_start = std::chrono::high_resolution_clock::now();
    Tensor *x, *y;
    bool batch_loaded = datagen.PopBatch(x, y);
    const auto load_end = std::chrono::high_resolution_clock::now();
    load_time += std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count();
    if (!batch_loaded) {
      std::cerr << "Error! The batch number " << b << " failed to load!" << std::endl;
      std::abort();
    }

    // Perform evaluation
    const auto eval_start = std::chrono::high_resolution_clock::now();
    eddl::eval_batch(model, {x}, {y});
    const auto eval_end = std::chrono::high_resolution_clock::now();
    eval_time += std::chrono::duration_cast<std::chrono::microseconds>(eval_end - eval_start).count();

    // Get the current losses and metrics
    const float curr_loss = eddl::get_losses(model)[0];
    const float curr_acc = eddl::get_metrics(model)[0];

    // Show current stats
    std::cout << "Testing: Batch " << b << "/" << n_te_batches << ": ";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Metrics[ loss=" << curr_loss << ", acc=" << curr_acc << " ]";
    std::cout << " - Timers[ ";
    std::cout << "avg_load_batch=" << (load_time / b)  * 1e-6 << "s";
    std::cout << ", avg_eval_batch=" << (eval_time / b) * 1e-6 << "s ]";
    std::cout << " - DataGenerator[ |fifo| = " << datagen.Size() << " ]";
    std::cout << std::endl;

    // Free memory for the next batch;
    delete x;
    delete y;
  }
  datagen.Stop();
  const auto test_end = std::chrono::high_resolution_clock::now();
  const float test_time = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_start).count();
  std::cout << "Test time elapsed = " << test_time * 1e-6 << "s\n\n";

  return TestResults(loss, acc);
}
