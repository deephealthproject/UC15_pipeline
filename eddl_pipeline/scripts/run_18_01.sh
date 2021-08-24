#!/bin/bash

img_type="RGB"
yaml_filename="${HOME}/data/deephealth/bimcv/covid19/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19-rgb.yaml"
#img_type="gray"
#yaml_filename="${HOME}/data/deephealth/bimcv/covid19/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19-gray.yaml"

scripts/train.sh --yaml_path ${yaml_filename} \
                --rgb_or_gray ${img_type} \
                --gpus 1,1 --lsb 10 \
                --model model_2a \
                --classifier_output sigmoid \
                --augmentations 2.2 \
                --epochs 1000 \
                --batch_size 10 \
                --optimizer SGD --learning_rate 1.0e-3 \
                --target_shape 1024,1024 \
                --workers 7
