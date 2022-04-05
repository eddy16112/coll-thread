#ifndef COLL_H
#define COLL_H

//#define COLL_USE_MPI

#define DEBUG_PRINT

#if defined(COLL_USE_MPI)
#include <mpi.h>
#else
#include <mpi.h>
#include <stdbool.h>
#define MAX_NB_THREADS 128
#endif

typedef struct local_buffer_s {
  void* buffers[MAX_NB_THREADS];
  bool buffers_ready[MAX_NB_THREADS];
} local_buffer_t;

typedef struct Coll_Comm_s {
#if defined(COLL_USE_MPI)
  MPI_Comm comm;
#else
  MPI_Comm comm;
  volatile local_buffer_t *local_buffer;
#endif
  int mpi_comm_size;
  int nb_threads;
  int mpi_rank;
  int tid;
} Coll_Comm;

extern local_buffer_t local_buffer;

#if defined(COLL_USE_MPI)
int MPI_Alltoall_thread(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                        void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                        Coll_Comm global_comm);
#else
int Coll_Alltoall_local(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                        void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                        Coll_Comm global_comm);
#endif
#endif // ifndef COLL_H