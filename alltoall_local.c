
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "coll.h"

#define ALLTOALL_USE_SENDRECV

local_buffer_t local_buffer;
 
int Coll_Alltoall_local(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                        void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                        Coll_Comm global_comm)
{	
  int res;
  MPI_Status status;

  assert(recvcount == sendcount);

  int total_size = global_comm.nb_threads;

  int sendtype_extent = 4;
  int recvtype_extent = 4;
 
  int global_rank = global_comm.tid;

  if (sendbuf == MPI_IN_PLACE) {
    assert(0);
  }

  global_comm.local_buffer = &local_buffer;
  global_comm.local_buffer->buffers[global_rank] = sendbuf;
  global_comm.local_buffer->buffers_ready[global_rank] = true;
  __sync_synchronize();

  int recvfrom_global_rank;
  int recvfrom_seg_id = global_rank;
  void *src_base = NULL;
	for(int i = 1 ; i < total_size + 1; i++) {
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    while (global_comm.local_buffer->buffers_ready[recvfrom_global_rank] != true);
    src_base = global_comm.local_buffer->buffers[recvfrom_global_rank];
    char* src = (char*)src_base + (ptrdiff_t)recvfrom_seg_id * sendtype_extent * sendcount;
    char* dst = (char*)recvbuf + (ptrdiff_t)recvfrom_global_rank * recvtype_extent * recvcount;
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, copy rank %d (seg %d, %p) to rank %d (seg %d, %p)\n", 
      i, global_rank, recvfrom_global_rank, recvfrom_seg_id, src, global_rank, recvfrom_global_rank, dst);
#endif
    memcpy(dst, src, sendcount * sendtype_extent);
	}

  return 0;
}