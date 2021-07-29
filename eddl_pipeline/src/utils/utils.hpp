#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <cxxopts.hpp>
#include <string>
#include <vector>

struct Arguments {
  std::string yaml_path; // Path to the yaml that defines the ECVL Dataset
  std::vector<int> target_shape; // Height and Width to resize the input
  int epochs;                    // Max number of training epochs
  int batch_size;        // Size of the batches to load (for training and test)
  std::vector<int> gpus; // Bit mask to select the GPUs to use
  int lsb;               // Number of batches between GPUs synchronization
  bool cpu;              // Use CPU computing device
  std::string mem_level; // Memory profile to use
  std::string augmentations; // DA version tag to use
  std::string model;         // Name of the topology to use
  std::string optimizer;     // Name of the training optimizer to use
  float learning_rate;       // Learning rate of the optimizer
  int seed;                  // Seed for the random computations
  std::string exp_path;      // Folder to store the experiments logs

  Arguments() = delete;
  Arguments(std::string yaml_path, const std::vector<int> &target_shape,
            const int epochs, const int batch_size,
            const std::vector<int> &gpus, const int lsb, const bool cpu,
            const std::string mem_level, const std::string augmentations,
            const std::string model, const std::string optimizer,
            const float learning_rate, const int seed,
            const std::string exp_path)
      : yaml_path(yaml_path), target_shape(target_shape), epochs(epochs),
        batch_size(batch_size), gpus(gpus), lsb(lsb), cpu(cpu),
        mem_level(mem_level), augmentations(augmentations), model(model),
        optimizer(optimizer), learning_rate(learning_rate), seed(seed),
        exp_path(exp_path) {}
};

Arguments parse_arguments(int argc, char **argv);

std::string get_current_time_str(const std::string time_format = "%d-%b_%H:%M");

#endif
