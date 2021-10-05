#!/bin/bash

#
# This script is for decompressing the downloaded .zip files of the dataset and to prepare it
# for the scripts of the pipeline.
#

print_help() {
    echo "Script Arguments:"
    echo "  -p FILE_PATH  Path to the dataset .zip file with the positive patients (default ./BIMCV-COVID19-cIter_1_2.zip)"
    echo "  -n FILE_PATH  Path to the dataset .zip file with the negative patients (default ./BIMCV-COVID19-cIter_1_2-Negative.zip)"
    echo "  -o DIR_PATH   Path to the folder to store the decompressed data (default ./)"
}

# Set default arguments
posi_path="./BIMCV-COVID19-cIter_1_2.zip"
neg_path="./BIMCV-COVID19-cIter_1_2-Negative.zip"
output_path="./"

# Parse script arguments
while getopts i:o:h flag
do
    case "${flag}" in
        p)
            posi_path=${OPTARG}
            if [[ ! -f $posi_path ]]; then
                echo "The input file path provided for positive patients is not valid!"
                exit 1
            elif [[ ! $posi_path = *.zip ]]; then
                echo "The input file provided for positive patients must be a .zip!"
                exit 1
            fi
            ;;
        n)
            neg_path=${OPTARG}
            if [[ ! -f $neg_path ]]; then
                echo "The input file path provided for negative patients is not valid!"
                exit 1
            elif [[ ! $neg_path = *.zip ]]; then
                echo "The input file provided for negative patients must be a .zip!"
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

#
# Unzip the two main zip files for positive and negative patients
#
echo "Decompressing the positive patients dataset zip file..."
unzip $posi_path -d $output_path
echo "Positives zip file decompressed!"

echo "Decompressing the negative patients dataset zip file..."
unzip $neg_path -d $output_path
echo "Negative zip file decompressed!"

#
# Prepare the positives dataset folder
#

# Go to the positives dataset folder
pushd $output_path/BIMCV-COVID19-cIter_1_2 > /dev/null

# Organize the dataset files
mkdir covid19_tars
mv *.tar.gz covid19_tars/
mkdir txt_list_files
mv *.tar-tvf.txt txt_list_files/

# Extract the full dataset from the tar.gz partitions
chmod +x 00_extract_data.sh
echo "Decompressing positives dataset .tar.gz partitions..."
./00_extract_data.sh # Script provided in the dataset files
echo "Positives partitions decompressed!"

# Go back to the working directory
popd > /dev/null

#
# Prepare the negatives dataset folder
#

# Go to the negatives dataset folder
pushd $output_path/BIMCV-COVID19-cIter_1_2-Negative > /dev/null

# Organize the dataset files
mkdir covid19_neg_tars
mv *.tar.gz covid19_neg_tars/
mkdir txt_list_files
mv *.tar-tvf.txt txt_list_files/

# Extract the full dataset from the tar.gz partitions
echo "Decompressing negatives dataset .tar.gz partitions..."
find covid19_neg_tars -name "*\.tar\.gz" -exec tar -xzf {} \;
echo "Negatives partitions decompressed!"

# Go back to the working directory
popd > /dev/null
