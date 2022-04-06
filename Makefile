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
COLL_SRC	+= coll.c
ifeq ($(strip $(COLL_NETWORKS)),mpi)
COLL_SRC	+= alltoall_thread.c \
						 gather_thread.c \
						 allgather_thread.c \
						 bcast_thread.c
endif
ifeq ($(strip $(COLL_NETWORKS)),local)
COLL_SRC	+= alltoall_local.c \
						 allgather_local.c
endif

COLL_OBJS	:= $(COLL_SRC:.c=.c.o)


COLL_TEST_SRC	+= alltoall_test.c \
						 		 gather_test.c \
								 allgather_test.c \
								 bcast_test.c

COLL_TEST_OBJS	:= $(COLL_TEST_SRC:.c=.c.o)

.PHONY: build clean

OUTFILE := alltoall_test gather_test allgather_test bcast_test

build: $(OUTFILE)

clean:
	rm -rf *.o $(OUTFILE)

$(COLL_OBJS) : %.c.o : %.c
	$(CC) -c -o $@ $< $(CC_FLAGS) $(INC_FLAGS)

$(COLL_TEST_OBJS) : %.c.o : %.c
	$(CC) -c -o $@ $< $(CC_FLAGS) $(INC_FLAGS)

alltoall_test: $(COLL_OBJS) alltoall_test.c.o
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

gather_test: $(COLL_OBJS) gather_test.c.o
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

allgather_test: $(COLL_OBJS) allgather_test.c.o
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

bcast_test: $(COLL_OBJS) bcast_test.c.o
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

# alltoall_thread2: alltoall_thread2.o
# 	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

