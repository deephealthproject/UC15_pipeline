#!/bin/bash

img_type="RGB"
yaml_filename="${HOME}/data/deephealth/bimcv/covid19/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19-rgb.yaml"
#img_type="gray"
#yaml_filename="${HOME}/data/deephealth/bimcv/covid19/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19-gray.yaml"

scripts/train.sh --yaml_path ${yaml_filename} \
                --rgb_or_gray ${img_type} \
                --gpus 0,1 --lsb 1 \
                --model model_2a \
                --classifier_output softmax \
                --augmentations 2.3 \
                --epochs 1000 \
                --batch_size 5 \
                --optimizer Adam --learning_rate 1.0e-4 \
                --target_shape 1024,1024 \
                --workers 7
