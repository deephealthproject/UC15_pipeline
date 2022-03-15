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
#SBATCH --time=01:00:00
#
# Name Nombre del trabajo
#SBATCH -J "EDDL"
#
# Partition cola donde se ejecutará. No utilizado
##SBATCH --partition=mpi

#Output fichero de salida (por defecto será slurm-numerodeltrabajo.out). No utilizado
##SBATCH --output=resnet50_1_nodo_2_inter.out

N=$1
BS=$2

img_type="RGB"
#yaml_filename="${HOME}/deephealth/datasets/winter_school/cropped/256x256/ecvl_256x256_normal-vs-covid.yaml"
yaml_filename="${HOME}/EDDL_yaml/winter-school/data/256x256/ecvl_256x256_normal-vs-covid.yaml"
#yaml_filename="${HOME}/EDDL_yaml/winter-school/data/256x256/split0/part.yaml"
yaml_folder="${HOME}/EDDL_yaml/winter-school/data/256x256/split"
yaml_file="part.yaml"

OUTPUT="distr_n${N}_bs$BS.out"

MPI_PARAM="-np $N -map-by node:PE=28 --report-bindings"

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
                 --epochs 10 \
                 --frozen_epochs 5 \
                 --batch_size $BS \
                 --optimizer Adam \
                 --learning_rate 1.0e-5 \
                 --lr_decay 0.01 \
                 --regularization l2 \
                 --regularization_factor 0.00001 \
                 --target_shape 256,256 \
                 --workers 6 
