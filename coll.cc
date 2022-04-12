#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include <pthread.h>

#include "coll.h"

#if defined (COLL_USE_MPI)
MPI_Datatype collChar = MPI_CHAR;
MPI_Datatype collInt = MPI_INT;
MPI_Datatype collFloat = MPI_FLOAT;
MPI_Datatype collDouble = MPI_DOUBLE;
#else
local_buffer_t local_buffer;

size_t get_dtype_size(collDataType_t dtype)
{
  if (dtype == collChar) {
    return sizeof(char);
  } else if (dtype == collInt) {
    return sizeof(int);
  } else if (dtype == collFloat) {
    return sizeof(float);
  } else if (dtype == collDouble) {
    return sizeof(double);
  } else {
    assert(0);
    return -1;
  }
} 
#endif

int Coll_Create_comm(Coll_Comm *global_comm, int global_comm_size, int global_rank, const int *mapping_table)
{
  global_comm->global_comm_size = global_comm_size;
  global_comm->global_rank = global_rank;
  global_comm->starting_tag = 0;
  global_comm->status = true;
#if defined(COLL_USE_MPI)
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  global_comm->mpi_comm_size = 1;
  global_comm->mpi_rank = mpi_rank;
  global_comm->comm = MPI_COMM_WORLD;
  if (mapping_table != NULL) {
    global_comm->mapping_table.global_rank = (int *)malloc(sizeof(int) * global_comm_size);
    global_comm->mapping_table.mpi_rank = (int *)malloc(sizeof(int) * global_comm_size);
    memcpy(global_comm->mapping_table.mpi_rank, mapping_table, sizeof(int) * global_comm_size);
    for (int i = 0; i < global_comm_size; i++) {
      global_comm->mapping_table.global_rank[i] = i;
    }
  }
#else
  global_comm->mpi_comm_size = 1;
  global_comm->mpi_rank = 0;
#endif
  return 0;
}

int Coll_Comm_free (Coll_Comm *global_comm)
{
#if defined(COLL_USE_MPI)
  if (global_comm->mapping_table.global_rank != NULL) {
    free(global_comm->mapping_table.global_rank);
    global_comm->mapping_table.global_rank = NULL;
  }
  if (global_comm->mapping_table.mpi_rank != NULL) {
    free(global_comm->mapping_table.mpi_rank);
    global_comm->mapping_table.mpi_rank = NULL;
  }
#endif
  global_comm->status = false;
  return 0;
}

int Coll_Alltoallv(const void *sendbuf, const int sendcounts[],
                   const int sdispls[], collDataType_t sendtype,
                   void *recvbuf, const int recvcounts[],
                   const int rdispls[], collDataType_t recvtype, 
                   Coll_Comm global_comm)
{
#if defined(COLL_USE_MPI)
  printf("MPI Alltoallv: global_rank %d, total_size %d\n", global_comm.global_rank, global_comm.global_comm_size);
  return Coll_Alltoallv_thread(sendbuf, sendcounts,
                               sdispls, sendtype,
                               recvbuf, recvcounts,
                               rdispls, recvtype, 
                               global_comm);
#else
  printf("Local Alltoallv: global_rank %d, total_size %d, send_buf %p\n", global_comm.global_rank, global_comm.global_comm_size, sendbuf);
  return Coll_Alltoallv_local(sendbuf, sendcounts,
                              sdispls, sendtype,
                              recvbuf, recvcounts,
                              rdispls, recvtype, 
                              global_comm);
#endif  
}

int Coll_Alltoall(void *sendbuf, int sendcount, collDataType_t sendtype, 
                  void *recvbuf, int recvcount, collDataType_t recvtype, 
                  Coll_Comm global_comm)
{
#if defined(COLL_USE_MPI)
  printf("MPI Alltoall: global_rank %d, total_size %d\n", global_comm.global_rank, global_comm.global_comm_size);
  return Coll_Alltoall_thread(sendbuf, sendcount, sendtype, 
                              recvbuf, recvcount, recvtype,
                              global_comm);
#else
  printf("Local Alltoall: global_rank %d, total_size %d, send_buf %p\n", global_comm.global_rank, global_comm.global_comm_size, sendbuf);
  return Coll_Alltoall_local(sendbuf, sendcount, sendtype, 
                             recvbuf, recvcount, recvtype,
                             global_comm);
#endif
}

int Coll_Gather(void *sendbuf, int sendcount, collDataType_t sendtype, 
                void *recvbuf, int recvcount, collDataType_t recvtype, 
                int root,
                Coll_Comm global_comm)
{
#if defined(COLL_USE_MPI)
  printf("MPI Gather: global_rank %d, total_size %d\n", global_comm.global_rank, global_comm.global_comm_size);
  return Coll_Gather_thread(sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype,
                            root,
                            global_comm);
#else
  printf("Local Gather: global_rank %d, total_size %d, send_buf %p\n", global_comm.global_rank, global_comm.global_comm_size, sendbuf);
  assert(0);
#endif  
}

int Coll_Allgather(void *sendbuf, int sendcount, collDataType_t sendtype, 
                   void *recvbuf, int recvcount, collDataType_t recvtype, 
                   Coll_Comm global_comm)
{
#if defined(COLL_USE_MPI)
  printf("MPI Allgather: global_rank %d, total_size %d\n", global_comm.global_rank, global_comm.global_comm_size);
  return Coll_Allgather_thread(sendbuf, sendcount, sendtype, 
                               recvbuf, recvcount, recvtype,
                               global_comm);
#else
  printf("Local Allgather: global_rank %d, total_size %d, send_buf %p\n", global_comm.global_rank, global_comm.global_comm_size, sendbuf);
  return Coll_Allgather_local(sendbuf, sendcount, sendtype, 
                              recvbuf, recvcount, recvtype,
                              global_comm);
#endif
}

int Coll_Bcast(void *buf, int count, collDataType_t type, 
               int root,
               Coll_Comm global_comm)
{
#if defined(COLL_USE_MPI)
  printf("MPI Bcast: global_rank %d, total_size %d\n", global_comm.global_rank, global_comm.mpi_comm_size * global_comm.nb_threads);
  return Coll_Bcast(buf, count, type, 
                    root,
                    global_comm);
#else
  printf("Local Bcast: global_rank %d, total_size %d, send_buf %p\n", global_comm.global_rank, global_comm.mpi_comm_size * global_comm.nb_threads, buf);
  assert(0);
#endif 
}
