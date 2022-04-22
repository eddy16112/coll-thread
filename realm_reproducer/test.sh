#!/bin/bash
set -x

NB_GPUS=4

IDX="$OMPI_COMM_WORLD_LOCAL_RANK"
GPUID=$(($IDX % $NB_GPUS))

#export CUDA_VISIBLE_DEVICES=$GPUID 
echo "rank $IDX, gpu $GPUID, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

#./test -ll:gpu 1 -dm:same_address_space 1 -b 4 -n 10240
./test -ll:gpu $NB_GPUS -dm:same_address_space 1 -b 4 -n 10240