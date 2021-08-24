#!/bin/bash

#img_type="RGB"
#yaml_filename="${HOME}/data/deephealth/bimcv/covid19/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19-rgb.yaml"
img_type="gray"
yaml_filename="${HOME}/data/deephealth/bimcv/covid19/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19-gray.yaml"

scripts/train.sh --yaml_path ${yaml_filename} \
                --rgb_or_gray ${img_type} \
                --gpus 0,1 --lsb 1 \
                --model model_3b \
                --classifier_output sigmoid \
                --augmentations 2.3 \
                --epochs 1000 \
                --batch_size 2 \
                --optimizer RMSprop --learning_rate 1.0e-5 \
                --target_shape 512,512 \
                --workers 7
