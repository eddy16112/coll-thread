
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "coll.h"
 
int Coll_Allgather_local(void *sendbuf, int sendcount, collDataType_t sendtype, 
                         void *recvbuf, int recvcount, collDataType_t recvtype, 
                         Coll_Comm global_comm)
{	
  assert(recvcount == sendcount);
  assert(sendtype == recvtype);

  int total_size = global_comm.global_comm_size;

  int sendtype_extent = get_dtype_size(sendtype);
  int recvtype_extent = get_dtype_size(recvtype);
  assert(sendtype_extent == recvtype_extent);

  int global_rank = global_comm.global_rank;

  if (sendbuf == recvbuf) {
    assert(0);
  }

  global_comm.local_buffer = &local_buffer;
  global_comm.local_buffer->buffers[global_rank] = sendbuf;
  global_comm.local_buffer->buffers_ready[global_rank] = true;
  __sync_synchronize();

  int recvfrom_global_rank;
	for(int i = 0 ; i < total_size; i++) {
    recvfrom_global_rank = i;
    while (global_comm.local_buffer->buffers_ready[recvfrom_global_rank] != true);
    char* src = (char*)global_comm.local_buffer->buffers[recvfrom_global_rank];
    char* dst = (char*)recvbuf + (ptrdiff_t)recvfrom_global_rank * recvtype_extent * recvcount;
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, dtype %d, copy rank %d (%p) to rank %d (%p)\n", 
      i, global_rank, sendtype_extent, recvfrom_global_rank, src, global_rank, dst);
#endif
    memcpy(dst, src, sendcount * sendtype_extent);
	}

  return 0;
}