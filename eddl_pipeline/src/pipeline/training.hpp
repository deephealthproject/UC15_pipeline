#ifndef _TRAINING_HPP_
#define _TRAINING_HPP_

#include <eddl/apis/eddl.h>
#include <string>
#include <vector>

#include <ecvl/support_eddl.h>

#include "../utils/utils.hpp"

struct TrainResults {
  // Metrics from each epoch
  std::vector<float> losses;
  std::vector<float> accs;
  std::vector<float> val_losses;
  std::vector<float> val_accs;

  std::string best_model_by_loss; // string path to the ONNX
  std::string best_model_by_acc;  // string path to the ONNX

  TrainResults(const std::vector<float> losses, std::vector<float> accs,
               std::vector<float> val_losses, std::vector<float> val_accs,
               std::string best_model_by_loss, std::string best_model_by_acc)
      : losses(losses), accs(accs), val_losses(val_losses), val_accs(val_accs),
        best_model_by_loss(best_model_by_loss),
        best_model_by_acc(best_model_by_acc) {}

  // Saves the training history (losses and accuracies) in a CSV file
  // in the path provided
  void save_hist_to_csv(const std::string &csv_path) const;
};

void dataset_summary(ecvl::DLDataset &dataset, const Arguments &args);

void apply_regularization(Net *model, const std::string &regularization, const float factor);

TrainResults train(ecvl::DLDataset &dataset, Net *model,
                   const std::string &exp_name, Arguments &args);

Optimizer *get_optimizer(const std::string &opt_name,
                         const float learning_rate);
#endif
