#ifndef _TEST_HPP_
#define _TEST_HPP_

#include <eddl/apis/eddl.h>
#include <string>
#include <vector>

#include <ecvl/support_eddl.h>

#include "../utils/utils.hpp"

struct TestResults {
  float loss;
  float acc;

  TestResults(const float loss, float acc) : loss(loss), acc(acc) {}
};

// Main function to use. It calls test_dataset() if args.use_dldataset is true
// else it calls test_datagen()
TestResults test(ecvl::DLDataset &dataset, Net *model, Arguments &args);

// Uses DLDataset to load the batches, no multithreading is used
TestResults test_dataset(ecvl::DLDataset &dataset, Net *model, Arguments &args);

// Using the multi threaded DataGenerator to load the batches
TestResults test_datagen(ecvl::DLDataset &dataset, Net *model, Arguments &args);
#endif
