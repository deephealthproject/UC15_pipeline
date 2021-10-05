# UC15 - BIMCV COVID 19+ pipeline
This repository contains the implementation of a Deep Learning pipeline to perform a classification task with chest X-ray images to detect COVID 19 cases and other pathologies. Here you can find the code developed for analyzing and preprocessing the dataset and also the code of the pipeline implemented with different DL libraries: EDDL (c++), PyEDDL and Pytorch. In the case of the EDDL and PyEDDL pipelines, the ECVL library has been used to load the batches of data to feed the models.

# Dataset
You can find the BIMCV COVID 19+ dataset [here](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/). Note that there are two versions: "Iteration 1" and "Iteration 1 + 2". Download the "Iteration 1 + 2" version.

The downloaded zip file weights **390 GB**. After decompressing the data, the size of the dataset is **780 GB**.

### How to decompress the dataset

We suggest you to use the bash script **decompress_dataset.sh** to decompress the downloaded zip file. To do so, run the comand:
```bash
./decompress_dataset.sh -i <PATH_TO_DATASET_ZIP> -o <OUTPUT_FOLDER>
```
Note: It will create a folder called **BIMCV-COVID19-cIter_1_2** inside the provided **OUTPUT_FOLDER**. The entire decompression may take a **couple of hours**.

# Requirements

To run the pipelines you need to have installed [PyECVL](https://github.com/deephealthproject/pyecvl) and [PyEDDL](https://github.com/deephealthproject/pyeddl).

Note: If you install the PyECVL using conda it will also install PyEDDL as a dependecy.

The aditional depencencies are listed in the **requirements.txt** file. You can install them using pip:

```bash
# Remember to activate your Python environment first (if you have it)
pip install -r requirements.txt
```

If you are going to try the Pytorch pipeline you will also need to install the dependencies of that pipeline:

```bash
pip install -r pytorch_requirements.txt
```

# Notebooks and Scripts

Inside the *pyeddl_pipeline* folder you can find all the notebooks and scripts developed to build the pipeline using EDDL and ECVL.

The *pytorch_pipeline* folder contains a replica of the pyeddl pipeline using Pytorch instead. Note that the notebooks used to analyze the data are only present in the *pyeddl_pipeline* folder.

#### **Important:** To run the Python scripts you must execute them from the corresponding pipeline folder in order to avoid problems when importing some Python modules.

Quick notebooks and scripts guide:

- **data_exploration.ipynb**: A notebook with a full dataset exploration. It analyzes the images, the available metadada and labels in order to get the most value out of them.
- **images_visualization.ipynb**: The aim of this notebook is tho show images. You can view random sampled images or you can select images by subject or session. It also shows the changes after applying some preprocessing to the images.
- **data_cleaning.ipynb**: This notebook provides an interface to easily label the samples as valid or not. The labels generated are stored in a TSV that can be passed to the "prepare_ecvl_dataset.py" script to only take the samples that are valid.
- **prepare_ecvl_dataset.py**: This script collects the needed data to prepare a YAML file for the ECVL to create a Dataset (ECVL object to load the data) for training the models. This script prepares the labels, aditional metadata and defines the dataset splits (train, validation, test).
- **train.py**: Main script to train the models. The experiments results (metrics, ONNX checkpoints...) will be stored in a new folder named "experiments".
- **test_ecvl_dataset.py**: A simple script to play with the ECVL DLDataset object and inspect the batches of data that are loaded.

Note: The Python scripts can be run with the **-h** flag to see possible configuration arguments. You probably need to use these arguments to designate where you have decompressed the dataset.

# C++ version of the pipeline

In the folder *eddl_pipeline* there is the C++ version of the pipeline. The main feature of these version is that it uses a custom dataloader class that can load the batches of data in parallel achieving a big speed up.

# Pytorch version of the pipeline

In order to compare the results and performance of the EDDL, another version of the pipeline has been developed using Pytorch. It's implemented in the *pytorch_pipeline* folder and has replicated many of the topologies implemented in the PyEDDL version to be able to compare the results.

# How to run

## 1. Prepare the data

Regardless of the pipeline version to be run (EDDL C++, PyEDDL or Pytorch), the *prepare_ecvl_dataset.py* must be executed to prepare the YAML file that the ECVL needs to load the data or the CSV file that the Pytorch version needs.
With the *prepare_ecvl_dataset.py* script you can modify the labels that you are goin to select for training and some other configurations for the data (use the *--help* flag to see all the options available).

To execute this script go to the *pyeddl_pipeline* folder and execute the script with the arguments configuration that you need. For example:

```bash
cd pyeddl_pipeline  # Needed to find some imports of the script
python prepare_ecvl_dataset.py --posi-path <PATH_TO_DATASET>/covid19_posi --neg-path <PATH_TO_DATASET>/covid19_neg --common-ids <PATH_TO_DATASET>/listjoin_ok.tsv
```
Note: *<PATH_TO_DATASET>* is the path to the folder where you decompressed the dataset with the script *decompress_dataset.sh*.

## 2. Train

### PyEDDL pipeline
```bash
# Inside pyeddl_pipeline folder
python train.py --yaml-path <PATH_TO_DATASET>/covid19_posi/ecvl_bimcv_covid19.yaml
```
Note: You can run *python train.py --help* to see all the flags available.

### EDDL C++ pipeline
```bash
# Inside eddl_pipeline folder
./scripts/compile.sh # Compile the pipeline and create the executable
./scripts/train.sh --yaml_path <PATH_TO_DATASET>/covid19_posi/ecvl_bimcv_covid19.yaml
```
Note: You can run *./script/train.sh --help* to see all the flags available.

### Pytorch pipeline
```bash
# Inside pytorch_pipeline folder
python train.py --data-tsv <PATH_TO_DATASET>/covid19_posi/ecvl_bimcv_covid19.tsv --labels 'normal' 'COVID 19'
```
Note: You can run *python train.py --help* to see all the flags available. The *--labels* flag is **IMPORTANT** to select the classes of interest because the tsv contains all the samples from all the classes.

In the case of the EDDL C++ and PyEDDL pipelines the results from the training script (models, metrics...) are stored in the *experiments* folder (It will be created automatically).
The Pytorch pipeline creates a *models_checkpoints* and a *logs* folders to store the results.
