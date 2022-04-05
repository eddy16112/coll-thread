DEBUG		?= 1
COLL_NETWORKS	?= mpi

CC			= mpicc
CC_FLAGS	?=
LD_FLAGS	?= -lpthread
INC_FLAGS	?=

CFLAGS		?=
LDFLAGS		?=
CC_FLAGS	+= $(CFLAGS)
LD_FLAGS	+= $(LDFLAGS)

ifeq ($(strip $(DEBUG)),1)
	CC_FLAGS	+= -g -O0
endif

ifeq ($(strip $(COLL_NETWORKS)),mpi)
	CC_FLAGS += -DCOLL_USE_MPI
endif

COLL_SRC	?=
COLL_SRC	+= coll.c	\
						 alltoall_test.c
ifeq ($(strip $(COLL_NETWORKS)),mpi)
COLL_SRC	+= alltoall_thread.c
endif
ifeq ($(strip $(COLL_NETWORKS)),local)
COLL_SRC	+= alltoall_local.c
endif



COLL_OBJS	:= $(COLL_SRC:.c=.c.o)

.PHONY: build clean

OUTFILE := alltoall_test

build: $(OUTFILE)

clean:
	rm -rf *.o $(OUTFILE)

$(COLL_OBJS) : %.c.o : %.c
	$(CC) -c -o $@ $< $(CC_FLAGS) $(INC_FLAGS)

alltoall_test: $(COLL_OBJS)
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

# alltoall_thread2: alltoall_thread2.o
# 	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

