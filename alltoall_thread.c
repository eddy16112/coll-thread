
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "coll.h"

#define ALLTOALL_USE_SENDRECV
 
int MPI_Alltoall_thread(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                        void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                        Coll_Comm global_comm)
{	
  int res;

  int total_size = global_comm.mpi_comm_size * global_comm.nb_threads;
	MPI_Status status;

  MPI_Aint lb, sendtype_extent, recvtype_extent;
  MPI_Type_get_extent(sendtype, &lb, &sendtype_extent);
  MPI_Type_get_extent(recvtype, &lb, &recvtype_extent);
 
  int global_rank = global_comm.mpi_rank * global_comm.nb_threads + global_comm.tid;

  void *sendbuf_tmp = NULL;
  if (sendbuf == MPI_IN_PLACE) {
    sendbuf_tmp = (void *)malloc(total_size * recvtype_extent * recvcount);
    memcpy(sendbuf_tmp, recvbuf, total_size * recvtype_extent * recvcount);
    // int * sendval = (int*)sendbuf_tmp;
    // printf("malloc %p, size %ld, [%d]\n", sendbuf_tmp, total_size * recvtype_extent * recvcount, sendval[0]);
  } else {
    sendbuf_tmp = sendbuf;
  }

#ifdef ALLTOALL_USE_SENDRECV
  int sendto_global_rank, recvfrom_global_rank, sendto_mpi_rank, recvfrom_mpi_rank;
	for(int i = 1 ; i < total_size + 1; i++) {
    sendto_global_rank  = (global_rank + i) % total_size;
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    char* src = (char*)sendbuf_tmp + (ptrdiff_t)sendto_global_rank * sendtype_extent * sendcount;
    char* dst = (char*)recvbuf + (ptrdiff_t)recvfrom_global_rank * recvtype_extent * recvcount;
    sendto_mpi_rank = sendto_global_rank / global_comm.nb_threads;
    recvfrom_mpi_rank = recvfrom_global_rank / global_comm.nb_threads;
    // tag: seg idx + rank_idx
    int send_tag = sendto_global_rank * 10000 + global_rank; // which dst seg it sends to (in dst rank)
    int recv_tag = global_rank * 10000 + recvfrom_global_rank; // idx of current seg we are receving (in src/my rank)
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, rank %d, tid %d, send %d to %d, send_tag %d, recv %d from %d, recv_tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, 
      sendto_global_rank, sendto_mpi_rank, send_tag, recvfrom_global_rank, recvfrom_mpi_rank, recv_tag);
#endif
    res = MPI_Sendrecv(src, sendcount, sendtype, sendto_mpi_rank, send_tag, dst, recvcount, recvtype, recvfrom_mpi_rank, recv_tag, global_comm.comm, &status);
    assert(res == MPI_SUCCESS);
	}
#elif defined (ALLTOALL_USE_SENDRECV_OLD)
    int dest_mpi_rank;
	for(int i = 0 ; i < total_size; i++) {
    char* src = (char*)sendbuf_tmp + i * sendtype_extent * sendcount;
    char* dst = (char*)recvbuf + i * recvtype_extent * recvcount;
    dest_mpi_rank = i / global_comm.nb_threads;
    int send_tag = i * 10000 + global_rank; // which seg it sends to
    int recv_tag = global_rank * 10000 + i; // idx of current seg
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, rank %d, tid %d, send %d to %d, send_tag %d, recv %d from %d, recv_tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, i, dest_mpi_rank, send_tag, i, dest_mpi_rank, recv_tag);
#endif
    res = MPI_Sendrecv(src, sendcount, sendtype, dest_mpi_rank, send_tag, dst, recvcount, recvtype, dest_mpi_rank, recv_tag, global_comm.comm, &status);
    assert(res == MPI_SUCCESS);
	}
#else
  int tag;
  int dest_mpi_rank;
	for(int i = 0 ; i < total_size; i++) {
    char* src = (char*)sendbuf_tmp + i * sendtype_extent * sendcount;
    dest_mpi_rank = i / global_comm.nb_threads;
    tag = i * 10000 + global_rank;
    res = MPI_Send(src, sendcount, sendtype, dest_mpi_rank, tag, global_comm.comm);
    assert(res == MPI_SUCCESS);
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, rank %d, tid %d, send %d to %d, send_tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, i, dest_mpi_rank, tag);
#endif
	}
  
  for(int i = 0 ; i < total_size; i++) {
    char* dst = (char*)recvbuf + i * recvtype_extent * recvcount;
    int dest_mpi_rank = i / global_comm.nb_threads;
    tag = global_rank * 10000 + i;
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, rank %d, tid %d, recv %d from %d, recv_tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, i, dest_mpi_rank, tag);
#endif
    res = MPI_Recv(dst, recvcount, recvtype, dest_mpi_rank, tag, global_comm.comm, &status);
    assert(res == MPI_SUCCESS);
	}
#endif

  if (sendbuf == MPI_IN_PLACE) {
    free(sendbuf_tmp);
  }

  return 0;
}