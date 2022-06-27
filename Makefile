DEBUG		?= 0
COLL_NETWORKS	?= mpi

ifeq ($(strip $(COLL_NETWORKS)),mpi)
CXX			= mpicxx
else
CXX = g++
endif
AR			= ar
CC_FLAGS	?=
LD_FLAGS	?= -lpthread
INC_FLAGS	?=
SO_FLAGS	?=
SHARED_OBJECTS ?= 1
COLL_LIBS := -Wl,-rpath . -L. -lcoll -lpthread

# CUDA_DIR = /usr/local/cuda-11.1
# GASNET_DIR=/scratch2/wwu/install/gasnet
# LEGION_DIR=/scratch2/wwu/legion-install-gasnet
# INC_FLAGS	+= -I$(LEGION_DIR)/include 
# LD_FLAGS	+= -L$(LEGION_DIR)/lib -lrealm -llegion -ldl -lrt -L$(GASNET_DIR)/lib -lgasnet-ibv-par -libverbs -L$(CUDA_DIR)/lib -lcuda

LEGION_DIR=/scratch2/wwu/legion-install/install
INC_FLAGS	+= -I$(LEGION_DIR)/include
LD_FLAGS	+= -L$(LEGION_DIR)/lib -lrealm -llegion -ldl -lrt

CFLAGS		?=
LDFLAGS		?=
CC_FLAGS	+= $(CFLAGS)
LD_FLAGS	+= $(LDFLAGS)

ifeq ($(strip $(SHARED_OBJECTS)),0)
SLIB_COLL     := libcoll.a
else
CC_FLAGS	  += -fPIC
ifeq ($(shell uname -s),Darwin)
SLIB_COLL     := libcoll.dylib
else
SLIB_COLL     := libcoll.so
endif
endif

ifeq ($(strip $(DARWIN)),1)
SO_FLAGS += -dynamiclib -single_module -undefined dynamic_lookup -fPIC
else
SO_FLAGS += -shared
endif

ifeq ($(strip $(DEBUG)),1)
	CC_FLAGS	+= -g -O0 -DDEBUG_LEGATE
else
	CC_FLAGS	+= -O2
endif

ifeq ($(strip $(COLL_NETWORKS)),mpi)
	CC_FLAGS += -DLEGATE_USE_GASNET
endif

COLL_SRC	?=
COLL_SRC	+= coll.cc
ifeq ($(strip $(COLL_NETWORKS)),mpi)
COLL_SRC	+= alltoall_thread_mpi.cc \
						 alltoallv_thread_mpi.cc \
						 gather_thread_mpi.cc \
						 allgather_thread_mpi.cc \
						 bcast_thread_mpi.cc
endif
ifeq ($(strip $(COLL_NETWORKS)),local)
COLL_SRC	+= alltoall_thread_local.cc \
						 alltoallv_thread_local.cc \
						 allgather_thread_local.cc
endif

COLL_OBJS	:= $(COLL_SRC:.cc=.cc.o)


COLL_TEST_SRC	+= alltoall_test.cc \
						 		 gather_test.cc \
								 allgather_test.cc \
								 bcast_test.cc \
								 alltoall_fake_sub_test.cc \
								 alltoallv_test.cc \
								 myalltoallv_test.cc \
								 alltoallv_con_test.cc \
								 alltoallv_con_test2.cc \
								 alltoallv_inplace_test.cc \
								 comm_test.cc \
								 alltoallv_test0.cc \

COLL_TEST_OBJS	:= $(COLL_TEST_SRC:.cc=.cc.o)

.PHONY: build clean

ifeq ($(strip $(COLL_NETWORKS)),mpi)
OUTFILE := alltoall_test gather_test allgather_test bcast_test alltoall_fake_sub_test alltoallv_test myalltoallv_test alltoallv_con_test alltoallv_con_test2 alltoallv_inplace_test alltoallv_test0
else
OUTFILE := alltoall_test allgather_test alltoall_fake_sub_test alltoallv_test myalltoallv_test alltoallv_con_test alltoallv_con_test2 alltoallv_inplace_test alltoallv_test0
endif


build: $(OUTFILE)

clean:
	rm -rf *.o *.so *.a $(OUTFILE)


ifeq ($(strip $(SHARED_OBJECTS)),0)
$(SLIB_COLL) : $(COLL_OBJS)
	rm -f $@
	$(AR) rcs $@ $^ $(LEGION_DIR)/lib/librealm.a $(LEGION_DIR)/lib/liblegion.a
else
$(SLIB_COLL) : $(COLL_OBJS)
	rm -f $@
	$(CXX) $(SO_FLAGS) -o $@ $(COLL_OBJS) $(LD_FLAGS)
endif

$(COLL_OBJS) : %.cc.o : %.cc
	$(CXX) -c -o $@ $< $(CC_FLAGS) $(INC_FLAGS)

$(COLL_TEST_OBJS) : %.cc.o : %.cc
	$(CXX) -c -o $@ $< $(CC_FLAGS) $(INC_FLAGS)

alltoall_test: $(SLIB_COLL) alltoall_test.cc.o
	$(CXX) -o $@ alltoall_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

gather_test: $(SLIB_COLL) gather_test.cc.o
	$(CXX) -o $@ gather_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

allgather_test: $(SLIB_COLL) allgather_test.cc.o
	$(CXX) -o $@ allgather_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

bcast_test: $(SLIB_COLL) bcast_test.cc.o
	$(CXX) -o $@ bcast_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

alltoall_fake_sub_test: $(SLIB_COLL) alltoall_fake_sub_test.cc.o
	$(CXX) -o $@ alltoall_fake_sub_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

alltoallv_test: $(SLIB_COLL) alltoallv_test.cc.o
	$(CXX) -o $@ alltoallv_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

myalltoallv_test: $(SLIB_COLL) myalltoallv_test.cc.o
	$(CXX) -o $@ myalltoallv_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

alltoallv_con_test: $(SLIB_COLL) alltoallv_con_test.cc.o
	$(CXX) -o $@ alltoallv_con_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

alltoallv_con_test2: $(SLIB_COLL) alltoallv_con_test2.cc.o
	$(CXX) -o $@ alltoallv_con_test2.cc.o $(CC_FLAGS) $(COLL_LIBS)

alltoallv_inplace_test: $(SLIB_COLL) alltoallv_inplace_test.cc.o
	$(CXX) -o $@ alltoallv_inplace_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

alltoallv_test0: $(SLIB_COLL) alltoallv_test0.cc.o
	$(CXX) -o $@ alltoallv_test0.cc.o $(CC_FLAGS) $(COLL_LIBS)

p2p_test: $(SLIB_COLL) p2p_test.cc.o
	$(CXX) -o $@ p2p_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

pingpong_test: $(SLIB_COLL) pingpong_test.cc.o
	$(CXX) -o $@ pingpong_test.cc.o $(CC_FLAGS) $(COLL_LIBS)

# alltoall_thread2: alltoall_thread2.o
# 	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

