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

  // Returns a string with the CSV representation of the history data of
  // losses and accuracies from the training phase
  std::string train_hist_csv_str() const;
};

void dataset_summary(ecvl::DLDataset &dataset, const Arguments &args);

// Main function to use. It calls train_dataset() if args.use_dldataset is true
// else it calls train_datagen()
TrainResults train(ecvl::DLDataset &dataset, Net *model,
                   const std::string &exp_name, Arguments &args);

// Uses DLDataset to load the batches, no multithreading is used
TrainResults train_dataset(ecvl::DLDataset &dataset, Net *model,
                           const std::string &exp_name, Arguments &args);

// Using the multi threaded DataGenerator to load the batches
TrainResults train_datagen(ecvl::DLDataset &dataset, Net *model,
                           const std::string &exp_name, Arguments &args);

Optimizer *get_optimizer(const std::string &opt_name,
                         const float learning_rate);
#endif
