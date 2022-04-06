
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "coll.h"

#define ALLGATHER_USE_BCAST
 
int Coll_Allgather_thread(void *sendbuf, int sendcount, collDataType_t sendtype, 
                          void *recvbuf, int recvcount, collDataType_t recvtype, 
                          Coll_Comm global_comm)
{	
  int res;

  int total_size = global_comm.mpi_comm_size * global_comm.nb_threads;
	MPI_Status status;
 
  int global_rank = global_comm.mpi_rank * global_comm.nb_threads + global_comm.tid;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    assert(0);
  }
  
#ifdef ALLGATHER_USE_BCAST
  global_comm.starting_tag = 0;
  Coll_Gather_thread(sendbuf, sendcount, sendtype, 
                     recvbuf, recvcount, recvtype, 
                     0, global_comm);

  global_comm.starting_tag = 1;
  Coll_Bcast_thread(recvbuf, recvcount * total_size, recvtype, 
                    0, global_comm);
#else
	for(int i = 0 ; i < total_size; i++) {
    // printf("global_rank %d, i %d\n", global_rank, i);
    global_comm.starting_tag = i;
    Coll_Gather_thread(sendbuf, sendcount, sendtype, 
                       recvbuf, recvcount, recvtype, 
                       i, global_comm);
	}
#endif
  return 0;
}