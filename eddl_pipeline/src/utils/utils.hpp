#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <cxxopts.hpp>
#include <string>

struct Arguments {
  std::string yaml_path;
  int epochs;
  int batch_size;
  std::string model;
  std::string optimizer;
  float learning_rate;
  int seed;

  Arguments() = delete;
  Arguments(std::string yaml_path, const int epochs, const int batch_size,
            const std::string model, const std::string optimizer,
            const float learning_rate, const int seed)
      : yaml_path(yaml_path), epochs(epochs), batch_size(batch_size),
        model(model), optimizer(optimizer), learning_rate(learning_rate),
        seed(seed) {}
};

Arguments parse_arguments(int argc, char **argv);

#endif
