#!/bin/bash

#
# This script is for decompressing the downloaded .zip file of the dataset and to prepare it
# for the scripts of the pipeline.
#

print_help() {
    echo "Script Arguments:"
    echo "  -i FILE_PATH  Path to the dataset .zip file (default ./bimcv_covid+.zip)"
    echo "  -o DIR_PATH   Path to the folder to store the decompressed data (default ./)"
}

# Set default arguments
input_path="./bimcv_covid+.zip"
output_path="./"

# Parse script arguments
while getopts i:o:h flag
do
    case "${flag}" in
        i)
            input_path=${OPTARG}
            if [[ ! -f $input_path ]]; then
                echo "The input file path provided is not valid!"
                exit 1
            elif [[ ! $input_path = *.zip ]]; then
                echo "The input file provided must be a .zip!"
                exit 1
            fi
            ;;
        o)
            output_path=${OPTARG}
            if [[ ! -d $output_path ]]; then
                echo "The output path provided is not valid!"
                exit 1
            fi
            ;;
        h) 
            print_help
            exit 0
            ;;
        \?)
            # Invalid argument detected
            print_help
            exit 1
            ;;
    esac
done

# Unzip main dataset file
echo "Decompressing dataset main file..."
unzip $input_path -d $output_path
echo "Dataset main file decompressed!"

# Go to dataset folder
cd $output_path/BIMCV-COVID19-cIter_1_2

# Organize the dataset files
mkdir covid19_tars
mv *.tar.gz covid19_tars/
mkdir txt_list_files
mv *.tar-tvf.txt txt_list_files/

# Extract the full dataset from the tar.gz partitions
chmod +x 00_extract_data.sh
echo "Decompressing dataset .tar.gz partitions..."
./00_extract_data.sh # Script provided in the dataset files
echo "Partitions decompressed!"
