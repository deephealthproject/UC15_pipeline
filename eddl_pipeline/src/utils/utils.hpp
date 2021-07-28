#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <cxxopts.hpp>
#include <string>
#include <vector>

struct Arguments {
  std::string yaml_path;
  std::vector<int> target_shape;
  int epochs;
  int batch_size;
  std::vector<int> gpus;
  int lsb;
  bool cpu;
  std::string mem_level;
  std::string augmentations;
  std::string model;
  std::string optimizer;
  float learning_rate;
  int seed;

  Arguments() = delete;
  Arguments(std::string yaml_path, const std::vector<int> &target_shape,
            const int epochs, const int batch_size,
            const std::vector<int> &gpus, const int lsb, const bool cpu,
            const std::string mem_level, const std::string augmentations,
            const std::string model, const std::string optimizer,
            const float learning_rate, const int seed)
      : yaml_path(yaml_path), target_shape(target_shape), epochs(epochs),
        batch_size(batch_size), gpus(gpus), lsb(lsb), cpu(cpu),
        mem_level(mem_level), augmentations(augmentations), model(model),
        optimizer(optimizer), learning_rate(learning_rate), seed(seed) {}
};

Arguments parse_arguments(int argc, char **argv);

std::string get_current_time_str(const std::string time_format = "%d-%b_%H:%M");

#endif
