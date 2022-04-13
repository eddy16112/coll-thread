#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include <pthread.h>

#include "coll.h"

#if defined (LEGATE_USE_GASNET)
MPI_Datatype collChar = MPI_CHAR;
MPI_Datatype collInt = MPI_INT;
MPI_Datatype collFloat = MPI_FLOAT;
MPI_Datatype collDouble = MPI_DOUBLE;
#else
local_buffer_t local_buffer[BUFFER_SWAP_SIZE];

static pthread_barrier_t local_barrier;

static bool coll_local_inited = false;

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

int Coll_Create_comm(collComm_t global_comm, int global_comm_size, int global_rank, const int *mapping_table)
{
  global_comm->global_comm_size = global_comm_size;
  global_comm->global_rank = global_rank;
  global_comm->starting_tag = 0;
  global_comm->status = true;
#if defined(LEGATE_USE_GASNET)
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
  global_comm->current_buffer_idx = 0;
  // for (int i = 0; i < BUFFER_SWAP_SIZE; i++) {
  //   for (int j = 0; j < MAX_NB_THREADS; j++) {
  //     global_comm->local_buffer->buffers_ready[i][j] = false;
  //   }
  // }
#endif
  return 0;
}

int Coll_Comm_free (collComm_t global_comm)
{
#if defined(LEGATE_USE_GASNET)
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
                   collComm_t global_comm)
{
#if defined(LEGATE_USE_GASNET)
  printf("MPI Alltoallv: global_rank %d, total_size %d\n", global_comm->global_rank, global_comm->global_comm_size);
  return Coll_Alltoallv_thread(sendbuf, sendcounts,
                               sdispls, sendtype,
                               recvbuf, recvcounts,
                               rdispls, recvtype, 
                               global_comm);
#else
  printf("Local Alltoallv: global_rank %d, total_size %d, send_buf %p\n", global_comm->global_rank, global_comm->global_comm_size, sendbuf);
  return Coll_Alltoallv_local(sendbuf, sendcounts,
                              sdispls, sendtype,
                              recvbuf, recvcounts,
                              rdispls, recvtype, 
                              global_comm);
#endif  
}

int Coll_Alltoall(void *sendbuf, int sendcount, collDataType_t sendtype, 
                  void *recvbuf, int recvcount, collDataType_t recvtype, 
                  collComm_t global_comm)
{
#if defined(LEGATE_USE_GASNET)
  printf("MPI Alltoall: global_rank %d, total_size %d\n", global_comm->global_rank, global_comm->global_comm_size);
  return Coll_Alltoall_thread(sendbuf, sendcount, sendtype, 
                              recvbuf, recvcount, recvtype,
                              global_comm);
#else
  printf("Local Alltoall: global_rank %d, total_size %d, send_buf %p\n", global_comm->global_rank, global_comm->global_comm_size, sendbuf);
  return Coll_Alltoall_local(sendbuf, sendcount, sendtype, 
                             recvbuf, recvcount, recvtype,
                             global_comm);
#endif
}

int Coll_Gather(void *sendbuf, int sendcount, collDataType_t sendtype, 
                void *recvbuf, int recvcount, collDataType_t recvtype, 
                int root,
                collComm_t global_comm)
{
#if defined(LEGATE_USE_GASNET)
  printf("MPI Gather: global_rank %d, total_size %d\n", global_comm->global_rank, global_comm->global_comm_size);
  return Coll_Gather_thread(sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype,
                            root,
                            global_comm);
#else
  printf("Local Gather: global_rank %d, total_size %d, send_buf %p\n", global_comm->global_rank, global_comm->global_comm_size, sendbuf);
  assert(0);
#endif  
}

int Coll_Allgather(void *sendbuf, int sendcount, collDataType_t sendtype, 
                   void *recvbuf, int recvcount, collDataType_t recvtype, 
                   collComm_t global_comm)
{
#if defined(LEGATE_USE_GASNET)
  printf("MPI Allgather: global_rank %d, total_size %d\n", global_comm->global_rank, global_comm->global_comm_size);
  return Coll_Allgather_thread(sendbuf, sendcount, sendtype, 
                               recvbuf, recvcount, recvtype,
                               global_comm);
#else
  printf("Local Allgather: global_rank %d, total_size %d, send_buf %p\n", global_comm->global_rank, global_comm->global_comm_size, sendbuf);
  return Coll_Allgather_local(sendbuf, sendcount, sendtype, 
                              recvbuf, recvcount, recvtype,
                              global_comm);
#endif
}

int Coll_Bcast(void *buf, int count, collDataType_t type, 
               int root,
               collComm_t global_comm)
{
#if defined(LEGATE_USE_GASNET)
  printf("MPI Bcast: global_rank %d, total_size %d\n", global_comm->global_rank, global_comm->mpi_comm_size * global_comm->nb_threads);
  return Coll_Bcast(buf, count, type, 
                    root,
                    global_comm);
#else
  printf("Local Bcast: global_rank %d, total_size %d, send_buf %p\n", global_comm->global_rank, global_comm->mpi_comm_size * global_comm->nb_threads, buf);
  assert(0);
#endif 
}

#ifndef LEGATE_USE_GASNET
void Coll_Update_buffer(collComm_t global_comm)
{
  global_comm->current_buffer_idx ++;
  global_comm->current_buffer_idx %= BUFFER_SWAP_SIZE;
  // printf("rank %d, buffer idx %d\n", global_comm->global_rank, global_comm->current_buffer_idx);
}

// called from main thread
void Coll_init_local(int nb_threads)
{
  for (int i = 0; i < BUFFER_SWAP_SIZE; i++) {
    local_buffer_t *buffer = &(local_buffer[i]);
    for (int j = 0; j < MAX_NB_THREADS; j++) {
      buffer->buffers[j] = NULL;
      buffer->displs[j] = NULL;
      buffer->buffers_ready[j] = false;
    }
  }

  pthread_barrier_init(&local_barrier, NULL, nb_threads);

  coll_local_inited = true;
}

void Coll_finalize_local(void)
{
  assert(coll_local_inited == true);
  pthread_barrier_destroy(&local_barrier);
  coll_local_inited = false;
}

void Coll_barrier_local(void)
{
  assert(coll_local_inited == true);
  pthread_barrier_wait(&local_barrier);
}
#endif