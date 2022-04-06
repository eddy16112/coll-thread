#ifndef COLL_H
#define COLL_H

//#define DEBUG_PRINT

#if defined (COLL_USE_MPI)
#include <mpi.h>
#else
#include <stdbool.h>
#define MAX_NB_THREADS 128

typedef struct local_buffer_s {
  void* buffers[MAX_NB_THREADS];
  bool buffers_ready[MAX_NB_THREADS];
} local_buffer_t;

extern local_buffer_t local_buffer;

#endif

#if defined (COLL_USE_MPI)
typedef MPI_Datatype collDataType_t;
// TODO: fix it
extern MPI_Datatype collChar;
extern MPI_Datatype collInt;
extern MPI_Datatype collFloat;
extern MPI_Datatype collDouble;
#else // NCCL and local
typedef enum { 
  collChar       = 0,
  collInt        = 2,
  collUint32     = 3,
  collInt64      = 4,
  collUint64     = 5,
#if 0
  collHalf       = 6,
#endif  
  collFloat      = 7,
  collDouble     = 8,
} collDataType_t;
#endif

typedef struct Coll_Comm_s {
#if defined (COLL_USE_MPI)
  MPI_Comm comm;
#else
  volatile local_buffer_t *local_buffer;
#endif
  int mpi_comm_size;
  int nb_threads;
  int mpi_rank;
  int tid;
  int starting_tag;
} Coll_Comm;

int Coll_Alltoall(void *sendbuf, int sendcount, collDataType_t sendtype, 
                  void *recvbuf, int recvcount, collDataType_t recvtype, 
                  Coll_Comm global_comm);

int Coll_Gather(void *sendbuf, int sendcount, collDataType_t sendtype, 
                void *recvbuf, int recvcount, collDataType_t recvtype, 
                int root,
                Coll_Comm global_comm);

int Coll_Allgather(void *sendbuf, int sendcount, collDataType_t sendtype, 
                   void *recvbuf, int recvcount, collDataType_t recvtype, 
                   Coll_Comm global_comm);

int Coll_Bcast(void *buf, int count, collDataType_t type, 
               int root,
               Coll_Comm global_comm);

#if defined (COLL_USE_MPI)
int Coll_Alltoall_thread(void *sendbuf, int sendcount, collDataType_t sendtype, 
                        void *recvbuf, int recvcount, collDataType_t recvtype, 
                        Coll_Comm global_comm);

int Coll_Gather_thread(void *sendbuf, int sendcount, collDataType_t sendtype, 
                       void *recvbuf, int recvcount, collDataType_t recvtype, 
                       int root,
                       Coll_Comm global_comm);

int Coll_Allgather_thread(void *sendbuf, int sendcount, collDataType_t sendtype, 
                          void *recvbuf, int recvcount, collDataType_t recvtype, 
                          Coll_Comm global_comm);

int Coll_Bcast_thread(void *buf, int count, collDataType_t type, 
                      int root,
                      Coll_Comm global_comm);
#else
size_t get_dtype_size(collDataType_t dtype);
int Coll_Alltoall_local(void *sendbuf, int sendcount, collDataType_t sendtype, 
                        void *recvbuf, int recvcount, collDataType_t recvtype, 
                        Coll_Comm global_comm);

int Coll_Allgather_local(void *sendbuf, int sendcount, collDataType_t sendtype, 
                         void *recvbuf, int recvcount, collDataType_t recvtype, 
                         Coll_Comm global_comm);
#endif
#endif // ifndef COLL_H