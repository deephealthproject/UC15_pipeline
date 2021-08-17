#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <cxxopts.hpp>
#include <ostream>
#include <string>
#include <vector>

struct Arguments {
  std::string yaml_path;         // Path to the yaml that defines the ECVL Dataset
  std::vector<int> target_shape; // Height and Width to resize the input
  std::string rgb_or_gray;       // Whether to use RGB or Gray images
  int epochs;                    // Max number of training epochs
  int batch_size;                // Size of the batches to load (for training and test)
  bool use_dldataset;            // Use DLDataset to load the batches (not DataGenerator)
  int workers;                   // Number of workers threads in the DataGenerator
  std::vector<int> gpus;         // Bit mask to select the GPUs to use
  int lsb;                       // Number of batches between GPUs synchronization
  bool cpu;                      // Use CPU computing device
  std::string mem_level;         // Memory profile to use
  std::string augmentations;     // DA version tag to use
  std::string model;             // Name of the topology to use
  std::string ckpt;              // ONNX file to use as checkpoint to start training
  std::string optimizer;         // Name of the training optimizer to use
  float learning_rate;           // Learning rate of the optimizer
  int seed;                      // Seed for the random computations
  std::string exp_path;          // Folder to store the experiments logs
  std::string classifier_output; // Activation type at output: softmax or sigmoid 

  Arguments() = delete;
  Arguments(std::string yaml_path,
            const std::vector<int> &target_shape,
            const std::string rgb_or_gray,
            const int epochs,
            const int batch_size,
            const bool use_dldataset,
            const int workers,
            const std::vector<int> &gpus,
            const int lsb,
            const bool cpu,
            const std::string mem_level,
            const std::string augmentations,
            const std::string model,
            const std::string ckpt,
            const std::string optimizer,
            const float learning_rate,
            const int seed,
            const std::string exp_path,
            const std::string classifier_output)
      : yaml_path(yaml_path),
        target_shape(target_shape),
        rgb_or_gray(rgb_or_gray),
        epochs(epochs),
        batch_size(batch_size),
        use_dldataset(use_dldataset),
        workers(workers),
        gpus(gpus),
        lsb(lsb),
        cpu(cpu),
        mem_level(mem_level),
        augmentations(augmentations),
        model(model),
        ckpt(ckpt),
        optimizer(optimizer),
        learning_rate(learning_rate),
        seed(seed),
        exp_path(exp_path),
        classifier_output(classifier_output)
    {
    }
};

// Writes the str representation of the Arguments class in json format
std::ostream &operator<<(std::ostream &out, Arguments args);

Arguments parse_arguments(int argc, char **argv);

std::string get_current_time_str(const std::string time_format = "%d-%b_%H:%M");

#endif
