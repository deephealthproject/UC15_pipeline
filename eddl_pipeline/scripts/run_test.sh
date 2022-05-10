#!/bin/bash

img_type="RGB"
#yaml_filename="${HOME}/deephealth/datasets/winter_school/cropped/256x256/ecvl_256x256_normal-vs-covid.yaml"
yaml_filename="${HOME}/EDDL_yaml/winter-school/data/256x256/ecvl_256x256_normal-vs-covid.yaml"

# process arguments
while  [ $# -ge 2 ]
do
	case $1 in 
		-n) PROCS=$2 ; shift ;;
		-bs) BS=$2 ; shift ;;
		*) break ;;
	esac
	shift
done

OUTPUT="sequential-$BS.out"

time scripts/train.sh --yaml_path ${yaml_filename} \
                 --rgb_or_gray ${img_type} \
                 --gpus 1 \
                 --model Pretrained_ResNet101 \
                 --classifier_output sigmoid \
                 --augmentations 1.1 \
                 --epochs 10 \
                 --frozen_epochs 5 \
                 --batch_size $BS \
                 --optimizer Adam \
                 --learning_rate 1.0e-5 \
                 --lr_decay 0.01 \
                 --regularization l2 \
                 --regularization_factor 0.00001 \
                 --target_shape 256,256 \
                 --mpi_average 1 \
                 --workers 6 > $OUTPUT
