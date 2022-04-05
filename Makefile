DEBUG ?= 1

CC = mpicc
CC_FLAGS ?=
LD_FLAGS ?= -lpthread
INC_FLAGS ?=

CFLAGS ?=
LDFLAGS ?=
CC_FLAGS += $(CFLAGS)
LD_FLAGS += $(LDFLAGS)

ifeq ($(strip $(DEBUG)),1)
  CC_FLAGS	+= -g -O0
endif

.PHONY: build clean

OUTFILE := alltoall_test

build: $(OUTFILE)

clean:
	rm -rf *.o $(OUTFILE)

%.o: %.c
	$(CC) -c -o $@ $< $(CC_FLAGS) $(INC_FLAGS)

alltoall_test: alltoall_thread.o alltoall_local.o alltoall_test.o
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

# alltoall_thread2: alltoall_thread2.o
# 	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

