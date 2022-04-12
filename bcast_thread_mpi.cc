
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "coll.h"
 
int Coll_Bcast_thread(void *buf, int count, collDataType_t type, 
                      int root,
                      collComm_t global_comm)
{	
  int res;

  // int total_size = global_comm.mpi_comm_size * global_comm.nb_threads;
  int total_size = global_comm->global_comm_size;
	MPI_Status status;
 
  int global_rank = global_comm->global_rank;

  int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  assert(root == global_comm->mapping_table.global_rank[root]);

  int tag;

  assert(global_comm->starting_tag >= 0);
  
  // non-root
  if (global_rank != root) {
    tag = global_comm->starting_tag * 10000 + global_rank;
#ifdef DEBUG_PRINT
    printf("Bcast Recv global_rank %d, rank %d, tid %d, send to %d (%d), tag %d\n", 
      global_rank, global_comm.mpi_rank, global_comm.tid, 
      root, root_mpi_rank, tag);
#endif
    return MPI_Recv(buf, count, type, root_mpi_rank, tag, global_comm->comm, &status);
  } 

  // root
  int sendto_mpi_rank;
	for(int i = 0 ; i < total_size; i++) {
    sendto_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    assert(i == global_comm->mapping_table.global_rank[i]);
    tag = global_comm->starting_tag * 10000 + i;
#ifdef DEBUG_PRINT
    printf("Bcast i: %d === global_rank %d, rank %d, tid %d, send to %d (%d), tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, 
      i, sendto_mpi_rank, tag);
#endif
    if (global_rank != i) {
      res = MPI_Send(buf, count, type, sendto_mpi_rank, tag, global_comm->comm);
      assert(res == MPI_SUCCESS);
    }
	}

  return 0;
}