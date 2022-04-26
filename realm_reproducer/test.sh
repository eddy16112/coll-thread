#!/bin/bash
set -x

NB_GPUS=6

IDX="$OMPI_COMM_WORLD_LOCAL_RANK"
GPUID=$(($IDX % $NB_GPUS))
WORLD_SIZE="$OMPI_COMM_WORLD_SIZE"

#export CUDA_VISIBLE_DEVICES=$GPUID 
echo "rank $IDX, gpu $GPUID, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

#./test -ll:gpu 1 -ll:fsize 14000 -ll:util 2 -ll:bgwork 2 -dm:same_address_space 1 -b $WORLD_SIZE -n 204800 -lg:prof 8 -lg:prof_logfile prof_%.gz
#./test -ll:gpu 1 -ll:fsize 14000 -ll:util 2 -ll:bgwork 2 -dm:same_address_space 1 -b $WORLD_SIZE -n 204800 -i 3 -lg:prof 8 -lg:spy -logfile spy_%.log -lg:prof_logfile prof_%.gz
#./test -lg:local 0 -ll:cpu 4 -ll:gpu 1 -cuda:skipbusy -ll:util 2 -ll:bgwork 2 -ll:csize 4000 -ll:fsize 14000 -ll:zsize 32 -lg:eager_alloc_percentage 50 -dm:same_address_space 1 -b $WORLD_SIZE -n 20480000 -lg:prof 24 -lg:prof_logfile prof_%.gz
#./test -ll:gpu $NB_GPUS -ll:fsize 14000 -ll:util 2 -ll:bgwork 2 -dm:same_address_space 1 -b $(($NB_GPUS * $WORLD_SIZE)) -n 20480000 -lg:prof 2 -lg:prof_logfile prof_%.gz
#./test -ll:gpu $NB_GPUS -lg:local 0 -ll:cpu 4 -cuda:skipbusy -ll:util 2 -ll:bgwork 2 -ll:csize 4000 -ll:fsize 14000 -ll:zsize 32 -lg:eager_alloc_percentage 50 -dm:same_address_space 1 -b $(($NB_GPUS * $WORLD_SIZE)) -n 20480000 -lg:prof 4 -lg:prof_logfile prof_%.gz
#./test -lg:local 0 -ll:cpu 4 -ll:gpu 1 -cuda:skipbusy -ll:util 2 -ll:bgwork 2 -ll:csize 4000 -ll:fsize 14000 -ll:zsize 32 -lg:eager_alloc_percentage 50 -dm:same_address_space 1 -b $WORLD_SIZE -n 20480000
./test -ll:gpu $NB_GPUS -lg:local 0 -ll:cpu 4 -cuda:skipbusy -ll:util 2 -ll:bgwork 2 -ll:csize 4000 -ll:fsize 14000 -ll:zsize 32 -lg:eager_alloc_percentage 50 -dm:same_address_space 1 -b $(($NB_GPUS * $WORLD_SIZE)) -n 20480000
#./test -ll:gpu $NB_GPUS -ll:fsize 14000 -ll:util 2 -ll:bgwork 2 -dm:same_address_space 1 -b $(($NB_GPUS * $WORLD_SIZE)) -n 204800 -lg:prof 2 -lg:prof_logfile prof_%.gz
./test -ll:gpu 1 -ll:fsize 14000 -ll:util 2 -ll:bgwork 2 -dm:same_address_space 1 -b $WORLD_SIZE -n 20480000
#./test -ll:gpu $NB_GPUS -ll:fsize 14000 -ll:util 2 -ll:bgwork 2 -dm:same_address_space 1 -b $(($NB_GPUS * $WORLD_SIZE)) -n 20480000
