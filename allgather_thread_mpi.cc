
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "coll.h"

#define ALLGATHER_USE_BCAST
 
int Coll_Allgather_thread(void *sendbuf, int sendcount, collDataType_t sendtype, 
                          void *recvbuf, int recvcount, collDataType_t recvtype, 
                          collComm_t global_comm)
{	
  int total_size = global_comm->global_comm_size;

  MPI_Aint lb, sendtype_extent, recvtype_extent;
  MPI_Type_get_extent(sendtype, &lb, &sendtype_extent);
 
  int global_rank = global_comm->global_rank;

  void *sendbuf_tmp = NULL;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    sendbuf_tmp = (void *)malloc(sendtype_extent * sendcount);
    memcpy(sendbuf_tmp, recvbuf, sendtype_extent * sendcount);
    // int * sendval = (int*)sendbuf_tmp;
    // printf("malloc %p, size %ld, [%d]\n", sendbuf_tmp, total_size * recvtype_extent * recvcount, sendval[0]);
  } else {
    sendbuf_tmp = sendbuf;
  }
  
#ifdef ALLGATHER_USE_BCAST
  global_comm->starting_tag = 0;
  Coll_Gather_thread(sendbuf_tmp, sendcount, sendtype, 
                     recvbuf, recvcount, recvtype, 
                     0, global_comm);

  global_comm->starting_tag = 1;
  Coll_Bcast_thread(recvbuf, recvcount * total_size, recvtype, 
                    0, global_comm);
#else
  int global_rank = global_comm->mpi_rank * global_comm->nb_threads + global_comm->tid;
	for(int i = 0 ; i < total_size; i++) {
    // printf("global_rank %d, i %d\n", global_rank, i);
    global_comm.starting_tag = i;
    Coll_Gather_thread(sendbuf, sendcount, sendtype, 
                       recvbuf, recvcount, recvtype, 
                       i, global_comm);
	}
#endif

  if (sendbuf == recvbuf) {
    free(sendbuf_tmp);
  }
  
  return 0;
}