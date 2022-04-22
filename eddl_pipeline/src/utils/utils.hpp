#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <cxxopts.hpp>
#include <ostream>
#include <string>
#include <vector>

#include <eddl/apis/eddl.h>
#include <ecvl/support_eddl.h>

struct Arguments {
  std::string yaml_folder;       // Path to the yaml that defines the ECVL Dataset
  std::string yaml_name;         // Path to the yaml that defines the ECVL Dataset
  std::string yaml_path;         // Path to the yaml that defines the ECVL Dataset
  std::vector<int> target_shape; // Height and Width to resize the input
  std::string rgb_or_gray;       // Whether to use RGB or Gray images
  int epochs;                    // Max number of training epochs
  int frozen_epochs;             // Number of epochs before unfreezing the pretrained weights
  int batch_size;                // Size of the batches to load (for training and test)
  int workers;                   // Number of workers threads in the DataGenerator
  std::vector<int> gpus;         // Bit mask to select the GPUs to use
  int lsb;                       // Number of batches between GPUs synchronization
  bool cpu;                      // Use CPU computing device
  std::string mem_level;         // Memory profile to use
  std::string augmentations;     // DA version tag to use
  std::string model;             // Name of the topology to use
  std::string ckpt;              // ONNX file to use as checkpoint to start training
  std::string regularization;    // Adds the selected regularization type to all the layers of the model
  float regularization_factor;   // Factor for the selected regularization type
  std::string optimizer;         // Name of the training optimizer to use
  float learning_rate;           // Learning rate of the optimizer
  float lr_decay;                // Decay factor for the learning rate
  int seed;                      // Seed for the random computations
  int mpi_average;               // Initial nr of batches between avg_weights
  std::string exp_path;          // Folder to store the experiments logs
  std::string classifier_output; // Activation type at output: softmax or sigmoid 

  // The following attributes are assigned by the script (not the user)
  bool is_pretrained = false; // true is using a "Pretrained_" model
  std::vector<std::string> layers2init; // Name of the new layers added to the "Pretrained_" model
  std::vector<std::string> pretrained_layers; // Name of the original layers of the "Pretrained_" model

  Arguments() = delete;
  Arguments(std::string yaml_folder,
            std::string yaml_name,
            std::string yaml_path,
            const std::vector<int> &target_shape,
            const std::string rgb_or_gray,
            const int epochs,
            const int frozen_epochs,
            const int batch_size,
            const int workers,
            const std::vector<int> &gpus,
            const int lsb,
            const bool cpu,
            const std::string mem_level,
            const std::string augmentations,
            const std::string model,
            const std::string ckpt,
            const std::string regularization,
            const float regularization_factor,
            const std::string optimizer,
            const float learning_rate,
            const float lr_decay,
            const int seed,
            const std::string exp_path,
            const int mpi_average,
            const std::string classifier_output)
      : yaml_folder(yaml_folder),
        yaml_name(yaml_name),
        yaml_path(yaml_path),
        target_shape(target_shape),
        rgb_or_gray(rgb_or_gray),
        epochs(epochs),
        frozen_epochs(frozen_epochs),
        batch_size(batch_size),
        workers(workers),
        gpus(gpus),
        lsb(lsb),
        cpu(cpu),
        mem_level(mem_level),
        augmentations(augmentations),
        model(model),
        ckpt(ckpt),
        regularization(regularization),
        regularization_factor(regularization_factor),
        optimizer(optimizer),
        learning_rate(learning_rate),
        lr_decay(lr_decay),
        seed(seed),
        exp_path(exp_path),
        mpi_average(mpi_average),
        classifier_output(classifier_output)
    {
    }
};

// Writes the str representation of the Arguments class in json format
std::ostream &operator<<(std::ostream &out, Arguments args);

Arguments parse_arguments(int argc, char **argv);

std::string get_current_time_str(const std::string time_format = "%d-%b_%H:%M");

ecvl::ColorType get_color_type(const std::string rgb_or_gray);

CompServ *get_computing_service(const Arguments args);

#endif
