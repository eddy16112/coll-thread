DEBUG		?= 1
COLL_NETWORKS	?= mpi

CC			= mpicxx
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
COLL_SRC	+= coll.cc
ifeq ($(strip $(COLL_NETWORKS)),mpi)
COLL_SRC	+= alltoall_thread_mpi.cc \
						 gather_thread_mpi.cc \
						 allgather_thread_mpi.cc \
						 bcast_thread_mpi.cc
endif
ifeq ($(strip $(COLL_NETWORKS)),local)
COLL_SRC	+= alltoall_thread_local.cc \
						 allgather_thread_local.cc
endif

COLL_OBJS	:= $(COLL_SRC:.cc=.cc.o)


COLL_TEST_SRC	+= alltoall_test.cc \
						 		 gather_test.cc \
								 allgather_test.cc \
								 bcast_test.cc

COLL_TEST_OBJS	:= $(COLL_TEST_SRC:.c=.c.o)

.PHONY: build clean

OUTFILE := alltoall_test gather_test allgather_test bcast_test

build: $(OUTFILE)

clean:
	rm -rf *.o $(OUTFILE)

$(COLL_OBJS) : %.cc.o : %.cc
	$(CC) -c -o $@ $< $(CC_FLAGS) $(INC_FLAGS)

$(COLL_TEST_OBJS) : %.cc.o : %.cc
	$(CC) -c -o $@ $< $(CC_FLAGS) $(INC_FLAGS)

alltoall_test: $(COLL_OBJS) alltoall_test.cc.o
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

gather_test: $(COLL_OBJS) gather_test.cc.o
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

allgather_test: $(COLL_OBJS) allgather_test.cc.o
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

bcast_test: $(COLL_OBJS) bcast_test.cc.o
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

# alltoall_thread2: alltoall_thread2.o
# 	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)
