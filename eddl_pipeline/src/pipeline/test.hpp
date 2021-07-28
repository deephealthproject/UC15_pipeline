#ifndef _TEST_HPP_
#define _TEST_HPP_

#include <eddl/apis/eddl.h>
#include <string>
#include <vector>

#include <ecvl/support_eddl.h>

#include "../utils/utils.hpp"

struct TestResults {
  std::vector<float> loss;
  std::vector<float> acc;

  TestResults(const std::vector<float> loss, std::vector<float> acc)
      : loss(loss), acc(acc) {}
};

TestResults test(ecvl::DLDataset &dataset, Net *model, Arguments &args);
#endif
