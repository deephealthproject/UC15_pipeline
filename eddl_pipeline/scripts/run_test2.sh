#!/bin/bash

img_type="gray"
#yaml_filename="${HOME}/deephealth/datasets/winter_school/cropped/256x256/ecvl_256x256_normal-vs-covid.yaml"
#yaml_filename="${HOME}/EDDL_yaml/winter-school/data/256x256/ecvl_256x256_normal-vs-covid.yaml"
yaml_filename="${HOME}/EDDL_yaml/covid-qu-ex/Lung_Segmentation_Data/Lung_Segmentation_Data/dataset.yaml"
#yaml_filename="/data/kaggle/covid-qu-ex/Lung_Segmentation_Data/Lung_Segmentation_Data/dataset-reduced.yaml"

BS=10
PROCS=1

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

time scripts/train_2.sh --yaml_path ${yaml_filename} \
                 --rgb_or_gray ${img_type} \
                 --model "covid19-qu-ex-02" \
                 --classifier_output softmax \
                 --gpus 1 \
                 --augmentations 0.0 \
                 --epochs 100 \
                 --frozen_epochs 0 \
                 --batch_size $BS \
                 --optimizer RMSprop \
                 --learning_rate 1.0e-4 \
                 --lr_decay 0.005 \
                 --regularization l2 \
                 --regularization_factor 0.00001 \
                 --target_shape 256,256 \
                 --mpi_average 1 \
                 --workers 6 #> $OUTPUT

#		 --ckpt experiments/2022-06-01-17-59-27_net-covid19-qu-ex-01_DA-1.1_opt-RMSprop_lr-0.000100/ckpts/2022-06-01-17-59-27_net-covid19-qu-ex-01_DA-1.1_opt-RMSprop_lr-0.000100_epoch-11_loss-100.168694_acc-0.726232.onnx \

