# Copyright 2022 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 1		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

USE_CYCLIC_MAPPING ?= 0

# Put the binary file name here
OUTFILE		?= alltoall 
# List all the application source files here
ifeq ($(strip $(USE_GASNET)),1)
GEN_SRC		?= alltoall2.cc	\
			   ../coll.cc \
			   ../alltoall_thread_mpi.cc \
			   ../alltoallv_thread_mpi.cc \
			   ../allgather_thread_mpi.cc \
			   ../gather_thread_mpi.cc \
			   ../bcast_thread_mpi.cc
else
GEN_SRC		?= alltoall.cc	\
			   ../coll.cc \
			   ../alltoall_thread_local.cc	\
			   ../alltoallv_thread_local.cc	\
			   ../allgather_thread_local.cc
endif
GEN_GPU_SRC	?=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
CC_FLAGS	?=
ifeq ($(strip $(USE_GASNET)),1)
CC_FLAGS	+= -DCOLL_USE_MPI
endif
ifeq ($(strip $(USE_CYCLIC_MAPPING)),1)
CC_FLAGS	+= -DCYCLIC_MAPPING
endif
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?=

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

