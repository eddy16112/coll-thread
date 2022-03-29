DEBUG ?= 0

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

OUTFILE := alltoall_thread

build: $(OUTFILE)

clean:
	rm -rf *.o $(OUTFILE)

%.o: %.c
	$(CC) -c -o $@ $< $(CC_FLAGS) $(INC_FLAGS)

alltoall_thread: alltoall_thread.o
	$(CC) -o $@ $^ $(CC_FLAGS) $(LD_FLAGS)

