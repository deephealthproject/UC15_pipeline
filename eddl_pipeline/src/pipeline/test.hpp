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

TestResults test(ecvl::DLDataset &dataset, Net *model, Arguments &args);
#endif
