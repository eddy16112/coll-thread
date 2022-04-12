
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "coll.h"
 
int Coll_Alltoall_local(void *sendbuf, int sendcount, collDataType_t sendtype, 
                        void *recvbuf, int recvcount, collDataType_t recvtype, 
                        collComm_t global_comm)
{	
  int res;

  assert(recvcount == sendcount);
  assert(sendtype == recvtype);

  int total_size = global_comm->global_comm_size;

  int sendtype_extent = get_dtype_size(sendtype);
  int recvtype_extent = get_dtype_size(recvtype);
 
  int global_rank = global_comm->global_rank;

  if (sendbuf == recvbuf) {
    assert(0);
  }

  global_comm->local_buffer = &(local_buffer[global_comm->current_buffer_idx]);
  global_comm->local_buffer->buffers[global_rank] = sendbuf;
  global_comm->local_buffer->buffers_ready[global_rank] = true;
  __sync_synchronize();

  int recvfrom_global_rank;
  int recvfrom_seg_id = global_rank;
  void *src_base = NULL;
	for(int i = 1 ; i < total_size + 1; i++) {
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    while (global_comm->local_buffer->buffers_ready[recvfrom_global_rank] != true);
    src_base = global_comm->local_buffer->buffers[recvfrom_global_rank];
    char* src = (char*)src_base + (ptrdiff_t)recvfrom_seg_id * sendtype_extent * sendcount;
    char* dst = (char*)recvbuf + (ptrdiff_t)recvfrom_global_rank * recvtype_extent * recvcount;
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, dtype %d, copy rank %d (seg %d, %p) to rank %d (seg %d, %p)\n", 
      i, global_rank, sendtype_extent, recvfrom_global_rank, recvfrom_seg_id, src, global_rank, recvfrom_global_rank, dst);
#endif
    memcpy(dst, src, sendcount * sendtype_extent);
	}

  Coll_Update_buffer(global_comm);
  __sync_synchronize();

  return 0;
}