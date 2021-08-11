#!/bin/bash

scripts/train.sh --yaml_path ~/data/deephealth/bimcv/covid19/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19.yaml \
                --gpus 0,1 \
                --model model_2b \
                --augmentations 2.1 \
                --epochs 300 \
                --batch_size 2 \
                --optimizer SGD --learning_rate 1.0e-4 \
                --target_shape 1024,1024 \
                --workers 5
