#include "test.hpp"
#include <chrono>

TestResults test(ecvl::DLDataset &dataset, Net *model, Arguments &args) {
  // Get test split info
  dataset.SetSplit(ecvl::SplitType::test);
  int n_te_samples = dataset.GetSplit().size();
  int n_te_batches = n_te_samples / args.batch_size;

  // Auxiliary tensors to load the batches
  Tensor *x = new Tensor({args.batch_size, dataset.n_channels_,
                          args.target_shape[0], args.target_shape[1]});
  Tensor *y = new Tensor({args.batch_size, static_cast<int>(dataset.classes_.size())});

  // Reset batch counter and shuffle all the data splits
  dataset.ResetAllBatches();

  // Reset the accumulated loss value
  eddl::reset_loss(model);

  // Validation phase
  float load_time = 0.f;
  float eval_time = 0.f;
  auto test_start = std::chrono::high_resolution_clock::now();
  std::cout << "\nTest phase:\n";
  dataset.SetSplit(ecvl::SplitType::test);
  for (int b = 1; b <= n_te_batches; ++b) {
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

    // Show current loss and metrics
    std::cout << " Batch ";
    eddl::print_loss(model, b-1);
    std::cout << "- Timers[";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "avg_load_batch=" << (load_time / b) * 1e-6 << "s";
    std::cout << " avg_eval_batch=" << (eval_time / b) * 1e-6 << "s]";
    std::cout << std::endl;
  }
  auto test_end = std::chrono::high_resolution_clock::now();
  float test_time = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_start).count();
  std::cout << "Test time elapsed = " << test_time * 1e-6 << "s\n";

  // Free batch memory
  delete x;
  delete y;

  return TestResults({}, {});
}
