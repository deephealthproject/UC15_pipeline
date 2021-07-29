#ifndef _TRAINING_HPP_
#define _TRAINING_HPP_

#include <eddl/apis/eddl.h>
#include <string>
#include <vector>

#include <ecvl/support_eddl.h>

#include "../utils/utils.hpp"

struct TrainResults {
  std::vector<float> loss;        // train loss
  std::vector<float> acc;         // train accuracy
  std::vector<float> val_loss;    // validation loss
  std::vector<float> val_acc;     // validation accuracy
  std::string best_model_by_loss; // string path to the ONNX
  std::string best_model_by_acc;  // string path to the ONNX

  TrainResults(const std::vector<float> loss, std::vector<float> acc,
               std::vector<float> val_loss, std::vector<float> val_acc,
               std::string best_model_by_loss, std::string best_model_by_acc)
      : loss(loss), acc(acc), val_loss(val_loss), val_acc(val_acc),
        best_model_by_loss(best_model_by_loss),
        best_model_by_acc(best_model_by_acc) {}
};

void dataset_summary(ecvl::DLDataset &dataset, const Arguments &args);

TrainResults train(ecvl::DLDataset &dataset, Net *model,
                   const std::string &exp_name, Arguments &args);

Optimizer *get_optimizer(const std::string &opt_name,
                         const float learning_rate);
#endif
