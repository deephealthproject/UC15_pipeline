#include "utils.hpp"
#include <iomanip>
#include <iostream>
#include <ctime>
#include <sstream>

std::ostream& operator<<(std::ostream &out, Arguments args) {
  // Auxiliary lambda to print string attributes
  auto print_str_attr = [&out](const std::string &k, const std::string &v) {
    out << "    \"" << k << "\": \"" << v << "\"";
  };
  // Auxiliary lambda to print numeric attributes
  auto print_num_attr = [&out](const std::string &k, const auto &v) {
    out << "    \"" << k << "\": " << v;
  };
  // Auxiliary lambda to print vector<int> attributes
  auto print_vec_attr = [&out](const std::string &k, const std::vector<int> &v) {
    out << "    \"" << k << "\": [";
    for (int i = 0; i < v.size(); ++i) {
      out << "\n        " << v[i]; // 8 spaces for double tabulation
      if (i != v.size() - 1)
        out << ",";
      else
       out << "\n";
    }
    out << "    ]";
  };
  // Print the Arguments info in json format
  out << "{\n";
  print_str_attr("yaml_folder", args.yaml_folder); out << ",\n";
  print_str_attr("yaml_name", args.yaml_name); out << ",\n";
  print_str_attr("yaml_path", args.yaml_path); out << ",\n";
  print_vec_attr("target_shape", args.target_shape); out << ",\n";
  print_str_attr("rgb_or_gray", args.rgb_or_gray); out << ",\n";
  print_num_attr("epochs", args.epochs); out << ",\n";
  print_num_attr("frozen_epochs", args.frozen_epochs); out << ",\n";
  print_num_attr("batch_size", args.batch_size); out << ",\n";
  print_num_attr("workers", args.workers); out << ",\n";
  print_vec_attr("gpus", args.gpus); out << ",\n";
  print_num_attr("lsb", args.lsb); out << ",\n";
  print_num_attr("cpu", args.cpu); out << ",\n";
  print_str_attr("mem_level", args.mem_level); out << ",\n";
  print_str_attr("augmentations", args.augmentations); out << ",\n";
  print_str_attr("model", args.model); out << ",\n";
  print_str_attr("ckpt", args.ckpt); out << ",\n";
  print_str_attr("regularization", args.regularization); out << ",\n";
  print_num_attr("regularization_factor", args.regularization_factor); out << ",\n";
  print_str_attr("optimizer", args.optimizer); out << ",\n";
  print_num_attr("learning_rate", args.learning_rate); out << ",\n";
  print_num_attr("lr_decay", args.lr_decay); out << ",\n";
  print_num_attr("seed", args.seed); out << ",\n";
  print_str_attr("exp_path", args.exp_path); out << ",\n";
  print_num_attr("mpi_average", args.mpi_average); out << ",\n"; 
  print_str_attr("classifier_output", args.classifier_output); out << "\n"; // Dont put "," in the last attr
  out << "}";
  return out;
}

Arguments parse_arguments(int argc, char **argv) {
  cxxopts::Options options(argv[0], "BIMCV COVID 19+ classification training");
  // Change the maximum number of columns of the parser output messages (the help message)
  options.set_width(160);
  // Declare the program arguments
  options.add_options()
    ("yaml_folder", "[DISTR] Path to the ECVL Dataset YAML folder", cxxopts::value<std::string>()->default_value(""))
    ("yaml_name", "[DISTR] Name of the ECVL Dataset YAML file", cxxopts::value<std::string>()->default_value("part.yml"))
    ("y,yaml_path", "Path to the ECVL Dataset YAML file", cxxopts::value<std::string>()->default_value("../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19.yaml"))
    ("t,target_shape", "Height and Width to resize the images", cxxopts::value<std::vector<int>>()->default_value("256,256"))
    ("rgb_or_gray", "Whether to use RGB or Gray images", cxxopts::value<std::string>()->default_value("gray"))
    ("e,epochs", "Number of training epochs", cxxopts::value<int>()->default_value("10"))
    ("frozen_epochs", "Number of epochs before unfreezing the pretrained weights", cxxopts::value<int>()->default_value("5"))
    ("b,batch_size", "Number of samples per minibatch", cxxopts::value<int>()->default_value("32"))
    ("w,workers", "Number of workers threads to load batches in the DataGenerator", cxxopts::value<int>()->default_value("1"))
    ("g,gpus", "GPUs to use. Selected with a bit mask: \"1,1\" for two gpus", cxxopts::value<std::vector<int>>()->default_value("1"))
    ("lsb", "Number of batches to process between GPUs synchronizations", cxxopts::value<int>()->default_value("1"))
    ("c,cpu", "Use computing service CPU", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("mem_level", "Amount of memory to use: low_mem, mid_mem, full_mem", cxxopts::value<std::string>()->default_value("full_mem"))
    ("a,augmentations", "Version of data augmentations to use", cxxopts::value<std::string>()->default_value("0.0"))
    ("m,model", "Name of the model to train", cxxopts::value<std::string>()->default_value("model_1"))
    ("ckpt", "Path to an ONNX file to use as starting point for training", cxxopts::value<std::string>()->default_value(""))
    ("regularization", "Adds the selected regularization type to all the layers of the model", cxxopts::value<std::string>()->default_value(""))
    ("regularization_factor", "Factor for the selected regularization type", cxxopts::value<float>()->default_value("0.0001"))
    ("o,optimizer", "Name of the training optimizer", cxxopts::value<std::string>()->default_value("Adam"))
    ("l,learning_rate", "Value of the learning rate", cxxopts::value<float>()->default_value("0.0001"))
    ("lr_decay", "Decay factor for the learning rate", cxxopts::value<float>()->default_value("0.0"))
    ("s,seed", "Seed value for random computations", cxxopts::value<int>()->default_value("27"))
    ("exp_path", "Path to the folder to store the experiments", cxxopts::value<std::string>()->default_value("experiments"))
    ("mpi_average", "Initial nr of batches between avg_weights", cxxopts::value<int>()->default_value("1"))
    ("classifier_output", "Whether to use softmax or sigmoid as the activation function of the output layer", cxxopts::value<std::string>()->default_value("softmax"))
    ("h,help", "Print usage");

  // Read arguments
  auto result = options.parse(argc, argv);

  // Handle the [-h, --help] argument
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  return Arguments(result["yaml_folder"].as<std::string>(),
                   result["yaml_name"].as<std::string>(),
                   result["yaml_path"].as<std::string>(),
                   result["target_shape"].as<std::vector<int>>(),
                   result["rgb_or_gray"].as<std::string>(),
                   result["epochs"].as<int>(),
                   result["frozen_epochs"].as<int>(),
                   result["batch_size"].as<int>(),
                   result["workers"].as<int>(),
                   result["gpus"].as<std::vector<int>>(),
                   result["lsb"].as<int>(),
                   result["cpu"].as<bool>(),
                   result["mem_level"].as<std::string>(),
                   result["augmentations"].as<std::string>(),
                   result["model"].as<std::string>(),
                   result["ckpt"].as<std::string>(),
                   result["regularization"].as<std::string>(),
                   result["regularization_factor"].as<float>(),
                   result["optimizer"].as<std::string>(),
                   result["learning_rate"].as<float>(),
                   result["lr_decay"].as<float>(),
                   result["seed"].as<int>(),
                   result["exp_path"].as<std::string>(),
                   result["mpi_average"].as<int>(),
                   result["classifier_output"].as<std::string>());
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

ecvl::ColorType get_color_type(const std::string rgb_or_gray) {
  if (rgb_or_gray == "RGB") {
    return ecvl::ColorType::RGB;
  } else if (rgb_or_gray == "gray" || rgb_or_gray == "grey") {
    return ecvl::ColorType::GRAY;
  } else {
    std::cerr << "\nERROR: non-recognized color/monochrome specification \""
              << rgb_or_gray << "\"\n\n";
    return ecvl::ColorType::GRAY;
  }
}

CompServ *get_computing_service(const Arguments args) {
  if (args.cpu)
    return eddl::CS_CPU(-1, "full_mem");
  else
    return eddl::CS_GPU(args.gpus, args.lsb, "full_mem");
}
