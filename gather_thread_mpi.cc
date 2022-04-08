
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "coll.h"
 
int Coll_Gather_thread(void *sendbuf, int sendcount, collDataType_t sendtype, 
                       void *recvbuf, int recvcount, collDataType_t recvtype, 
                       int root,
                       Coll_Comm global_comm)
{	
  int res;

  // int total_size = global_comm.mpi_comm_size * global_comm.nb_threads;
  int total_size = global_comm.global_comm_size;
	MPI_Status status;
 
  int global_rank = global_comm.global_rank;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    assert(0);
  }

  int root_mpi_rank = global_comm.mapping_table.mpi_rank[root];
  assert(root == global_comm.mapping_table.global_rank[root]);

  int tag;

  assert(global_comm.starting_tag >= 0);
  
  // non-root
  if (global_rank != root) {
    tag = global_comm.starting_tag * 10000 + global_rank;
#ifdef DEBUG_PRINT
    printf("Gather Send global_rank %d, rank %d, tid %d, send to %d (%d), tag %d\n", 
      global_rank, global_comm.mpi_rank, global_comm.tid, 
      root, root_mpi_rank, tag);
#endif
    return MPI_Send(sendbuf, sendcount, sendtype, root_mpi_rank, tag, global_comm.comm);
  } 

  // root
  MPI_Aint incr, lb, recvtype_extent;
  MPI_Type_get_extent(recvtype, &lb, &recvtype_extent);
  incr = recvtype_extent * (ptrdiff_t)recvcount;
  char *dst = (char*)recvbuf;
  int recvfrom_mpi_rank;
	for(int i = 0 ; i < total_size; i++) {
    recvfrom_mpi_rank = global_comm.mapping_table.mpi_rank[i];
    assert(i == global_comm.mapping_table.global_rank[i]);
    tag = global_comm.starting_tag * 10000 + i;
#ifdef DEBUG_PRINT
    printf("Gather i: %d === global_rank %d, rank %d, tid %d, recv %p, from %d (%d), tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, 
      dst, i, recvfrom_mpi_rank, tag);
#endif
    assert(dst != NULL);
    if (global_rank == i) {
      memcpy(dst, sendbuf, incr);
    } else {
      res = MPI_Recv(dst, recvcount, recvtype, recvfrom_mpi_rank, tag, global_comm.comm, &status);
      assert(res == MPI_SUCCESS);
    }
    dst += incr;
	}

  return 0;
}