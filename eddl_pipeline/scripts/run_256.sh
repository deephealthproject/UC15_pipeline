#!/bin/bash
#======
#
# Project/Account (use your own) el nombre que aparece en squeue
#SBATCH -A plopez            
#
# Number of MPI tasks cuantos procesos mpi vas a utilizar. No utilizado
##SBATCH -n 2
#
# Number of tasks per node
##SBATCH --tasks-per-node=1 cuantos procesos de mpi quieres por nodo. No utilizado
#
# Tiempo de ejecución máximo
#SBATCH --time=10:00:00
#
# Name Nombre del trabajo
#SBATCH -J "UC15-256+"
#
# Partition cola donde se ejecutará. No utilizado
##SBATCH --partition=mpi

#Output fichero de salida (por defecto será slurm-numerodeltrabajo.out). No utilizado
##SBATCH --output=resnet50_1_nodo_2_inter.out

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

MPIAVG=1

WIDTH=256
#WIDTH=512
HEIGHT=$WIDTH
TARGET_SHAPE="$WIDTH","$HEIGHT"
SIZE="$WIDTH"x"$HEIGHT"

img_type="RGB"
#yaml_filename="${HOME}/deephealth/datasets/winter_school/cropped/${SIZE}/ecvl_${SIZE}_normal-vs-covid.yaml"
yaml_filename="${HOME}/EDDL_yaml/winter-school/data/${SIZE}/ecvl_${SIZE}_normal-vs-covid.yaml"
yaml_folder="${HOME}/EDDL_yaml/winter-school/data/${SIZE}/split"
yaml_file="part.yaml"

NAME="nccl_n${PROCS}_${SIZE}_bs${BS}"
OUTPUT=${NAME}.out
ERR=${NAME}.err


MPI_PARAM="-np $PROCS -map-by node:PE=28 --report-bindings"

#mpirun $MPI_PARAM -mca pls_rsh_agent "ssh -X -n" xterm -hold -e  scripts/distr_train.sh \
time mpirun $MPI_PARAM  scripts/distr_train.sh \
                 --yaml_folder ${yaml_folder} \
                 --yaml_name ${yaml_file} \
                 --yaml_path ${yaml_filename} \
                 --rgb_or_gray ${img_type} \
                 --gpus 1 \
                 --model Pretrained_ResNet101 \
                 --classifier_output sigmoid \
                 --augmentations 1.1 \
                 --epochs 20 \
                 --frozen_epochs 5 \
                 --batch_size $BS \
                 --optimizer Adam \
                 --learning_rate 1.0e-5 \
                 --lr_decay 0.01 \
                 --regularization l2 \
                 --regularization_factor 0.00001 \
                 --target_shape ${TARGET_SHAPE} \
                 --mpi_average $MPIAVG \
                 --workers 6 > $OUTPUT 2> $ERR

