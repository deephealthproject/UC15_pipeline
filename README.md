# UC15 - BIMCV COVID 19+ pipeline
This repository contains all the code developed for analyzing and processing the data, and the Deep Learning pipelines implemented using ECVL and EDDL to perform a classification task with chest X-ray images to detect COVID 19 cases and other pathologies.

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

# Notebooks and Scripts

Inside the *python* folder you can find all the notebooks and scripts developed in Python. 

#### **Important:** To run the Python scripts you must execute them from the *python* folder in order to avoid problems when importing some Python modules.

Quick notebooks and scripts guide:

- **data_exploration.ipynb**: A notebook with a full dataset exploration. It analyzes the images, the available metadada and labels in order to get the most value out of them.
- **images_visualization.ipynb**: The aim of this notebook is tho show images. You can view random sampled images or you can select images by subject or session. It also shows the changes after applying some preprocessing to the images.
- **prepare_ecvl_dataset.py**: This script collects the needed data to prepare a YAML file for the ECVL to create a Dataset (ECVL object to load the data) for training the models. This script prepares the labels, aditional metadata and defines the dataset splits (train, validation, test).

Note: The Python scripts can be run with the **-h** flag to see possible configuration arguments. You probably need to use these arguments to designate where you have decompressed the dataset.

# The pipeline development is in progress...
