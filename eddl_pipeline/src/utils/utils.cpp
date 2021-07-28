#include "utils.hpp"
#include <iomanip>
#include <iostream>
#include <ctime>
#include <sstream>

Arguments parse_arguments(int argc, char **argv) {
  cxxopts::Options options(argv[0], "BIMCV COVID 19+ classification training");
  // Change the maximum number of columns of the parser output messages (the help message)
  options.set_width(160);
  // Declare the program arguments
  options.add_options()
    ("y,yaml_path", "Path to the ECVL Dataset YAML file", cxxopts::value<std::string>()->default_value("../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19.yaml"))
    ("t,target_shape", "Height and Width to resize the images", cxxopts::value<std::vector<int>>()->default_value("256,256"))
    ("e,epochs", "Number of training epochs", cxxopts::value<int>()->default_value("10"))
    ("b,batch_size", "Number of samples per minibatch", cxxopts::value<int>()->default_value("32"))
    ("g,gpus", "GPUs to use. Selected with a bit mask: \"1,1\" for two gpus", cxxopts::value<std::vector<int>>()->default_value("1"))
    ("lsb", "Number of batches to process between GPUs synchronizations", cxxopts::value<int>()->default_value("1"))
    ("c,cpu", "Use computing service CPU", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("mem_level", "Amount of memory to use: low_mem, mid_mem, full_mem", cxxopts::value<std::string>()->default_value("full_mem"))
    ("a,augmentations", "Version of data augmentations to use", cxxopts::value<std::string>()->default_value("0.0"))
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
                   result["target_shape"].as<std::vector<int>>(),
                   result["epochs"].as<int>(),
                   result["batch_size"].as<int>(),
                   result["gpus"].as<std::vector<int>>(),
                   result["lsb"].as<int>(),
                   result["cpu"].as<bool>(),
                   result["mem_level"].as<std::string>(),
                   result["augmentations"].as<std::string>(),
                   result["model"].as<std::string>(),
                   result["optimizer"].as<std::string>(),
                   result["learning_rate"].as<float>(),
                   result["seed"].as<int>());
}

std::string get_current_time_str(const std::string time_format) {
  // Get the time
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  // Build the string
  std::ostringstream oss;
  oss << std::put_time(&tm, time_format.c_str());
  return oss.str();
}
