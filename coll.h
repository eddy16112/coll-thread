#ifndef COLL_H
#define COLL_H

//#define DEBUG_PRINT

#include <stddef.h>

#if defined (LEGATE_USE_GASNET)
#include <mpi.h>
#else
#include <stdbool.h>
#define MAX_NB_THREADS 128
#define BUFFER_SWAP_SIZE 2

typedef struct local_buffer_s {
  void* buffers[MAX_NB_THREADS];
  int* displs[MAX_NB_THREADS];
  bool buffers_ready[MAX_NB_THREADS];
} local_buffer_t;

extern local_buffer_t local_buffer[BUFFER_SWAP_SIZE];

#endif

#if defined (LEGATE_USE_GASNET)
typedef MPI_Datatype collDataType_t;
// TODO: fix it
extern MPI_Datatype collChar;
extern MPI_Datatype collInt;
extern MPI_Datatype collFloat;
extern MPI_Datatype collDouble;

typedef struct mapping_table_s {
  int *mpi_rank; // just for verification
  int *global_rank;
} mapping_table_t;
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
#if defined (LEGATE_USE_GASNET)
  MPI_Comm comm;
  mapping_table_t mapping_table;
#else
  volatile local_buffer_t *local_buffer;
  int current_buffer_idx;
#endif
  int mpi_comm_size; // not used
  int nb_threads; // not used
  int mpi_rank;
  int tid; // not used
  int global_rank;
  int global_comm_size;
  int starting_tag;
  bool status;
} Coll_Comm;

typedef Coll_Comm* collComm_t;

int Coll_Create_comm(collComm_t global_comm, int global_comm_size, int global_rank, const int *mapping_table);

int Coll_Comm_free (collComm_t global_comm);

int Coll_Alltoallv(const void *sendbuf, const int sendcounts[],
                   const int sdispls[], collDataType_t sendtype,
                   void *recvbuf, const int recvcounts[],
                   const int rdispls[], collDataType_t recvtype, 
                   collComm_t global_comm);

int Coll_Alltoall(void *sendbuf, int sendcount, collDataType_t sendtype, 
                  void *recvbuf, int recvcount, collDataType_t recvtype, 
                  collComm_t global_comm);

int Coll_Gather(void *sendbuf, int sendcount, collDataType_t sendtype, 
                void *recvbuf, int recvcount, collDataType_t recvtype, 
                int root,
                collComm_t global_comm);

int Coll_Allgather(void *sendbuf, int sendcount, collDataType_t sendtype, 
                   void *recvbuf, int recvcount, collDataType_t recvtype, 
                   collComm_t global_comm);

int Coll_Bcast(void *buf, int count, collDataType_t type, 
               int root,
               collComm_t global_comm);

#if defined (LEGATE_USE_GASNET)
int Coll_Alltoallv_thread(const void *sendbuf, const int sendcounts[],
                          const int sdispls[], collDataType_t sendtype,
                          void *recvbuf, const int recvcounts[],
                          const int rdispls[], collDataType_t recvtype, 
                          collComm_t global_comm);

int Coll_Alltoall_thread(void *sendbuf, int sendcount, collDataType_t sendtype, 
                        void *recvbuf, int recvcount, collDataType_t recvtype, 
                        collComm_t global_comm);

int Coll_Gather_thread(void *sendbuf, int sendcount, collDataType_t sendtype, 
                       void *recvbuf, int recvcount, collDataType_t recvtype, 
                       int root,
                       collComm_t global_comm);

int Coll_Allgather_thread(void *sendbuf, int sendcount, collDataType_t sendtype, 
                          void *recvbuf, int recvcount, collDataType_t recvtype, 
                          collComm_t global_comm);

int Coll_Bcast_thread(void *buf, int count, collDataType_t type, 
                      int root,
                      collComm_t global_comm);
#else
size_t get_dtype_size(collDataType_t dtype);

int Coll_Alltoallv_local(const void *sendbuf, const int sendcounts[],
                         const int sdispls[], collDataType_t sendtype,
                         void *recvbuf, const int recvcounts[],
                         const int rdispls[], collDataType_t recvtype, 
                         collComm_t global_comm);

int Coll_Alltoall_local(void *sendbuf, int sendcount, collDataType_t sendtype, 
                        void *recvbuf, int recvcount, collDataType_t recvtype, 
                        collComm_t global_comm);

int Coll_Allgather_local(void *sendbuf, int sendcount, collDataType_t sendtype, 
                         void *recvbuf, int recvcount, collDataType_t recvtype, 
                         collComm_t global_comm);

void Coll_Update_buffer(collComm_t global_comm);

void Coll_init_local(int nb_threads);

void Coll_finalize_local(void);

void Coll_barrier_local(void);
#endif
#endif // ifndef COLL_H