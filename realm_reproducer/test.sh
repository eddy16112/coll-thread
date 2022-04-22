#!/bin/bash
set -x

NB_GPUS=4

IDX="$OMPI_COMM_WORLD_LOCAL_RANK"
GPUID=$(($IDX % $NB_GPUS))
WORLD_SIZE="$OMPI_COMM_WORLD_SIZE"

#export CUDA_VISIBLE_DEVICES=$GPUID 
echo "rank $IDX, gpu $GPUID, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

#./test -ll:gpu 1 -dm:same_address_space 1 -b 8 -n 10240 -lg:prof 8 -lg:prof_logfile prof_%.gz
#./test -ll:gpu $NB_GPUS -dm:same_address_space 1 -b 8 -n 10240 -lg:prof 2 -lg:prof_logfile prof_%.gz
#./test -ll:gpu 1 -dm:same_address_space 1 -b $WORLD_SIZE -n 102400
./test -ll:gpu $NB_GPUS -dm:same_address_space 1 -b $(($NB_GPUS * $WORLD_SIZE)) -n 10240