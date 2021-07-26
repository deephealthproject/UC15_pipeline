#include "utils.hpp"
#include <iostream>

Arguments parse_arguments(int argc, char **argv) {
  cxxopts::Options options(argv[0], "BIMCV COVID 19+ classification training");
  // Change the maximum number of columns of the parser output messages (the help message)
  options.set_width(160);
  // Declare the program arguments
  options.add_options()
    ("y,yaml_path", "Path to the ECVL Dataset YAML file", cxxopts::value<std::string>()->default_value("../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19.yaml"))
    ("e,epochs", "Number of training epochs", cxxopts::value<int>()->default_value("10"))
    ("b,batch_size", "Number of samples per minibatch", cxxopts::value<int>()->default_value("32"))
    ("m,model", "Name of the model to train", cxxopts::value<std::string>()->default_value("model_1"))
    ("o,optimizer", "Name of the training optimizer", cxxopts::value<std::string>()->default_value("Adam"))
    ("l,learning_rate", "Value of the learning rate", cxxopts::value<float>()->default_value("0.0001"))
    ("s,seed", "Seed value for random computations", cxxopts::value<int>()->default_value("27"))
    ("h,help", "Print usage");

  // Read arguments
  auto result = options.parse(argc, argv);

  // Handle the [-h, --help] argument
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  return Arguments(result["yaml_path"].as<std::string>(),
                   result["epochs"].as<int>(),
                   result["batch_size"].as<int>(),
                   result["model"].as<std::string>(),
                   result["optimizer"].as<std::string>(),
                   result["learning_rate"].as<float>(),
                   result["seed"].as<int>());
}
