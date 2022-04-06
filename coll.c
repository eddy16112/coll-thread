#include <stdio.h>
#include <assert.h>

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

int Coll_Alltoall(void *sendbuf, int sendcount, collDataType_t sendtype, 
                  void *recvbuf, int recvcount, collDataType_t recvtype, 
                  Coll_Comm global_comm)
{
#if defined(COLL_USE_MPI)
  printf("MPI: Thread %d, total_size %d\n", global_comm.tid, global_comm.mpi_comm_size * global_comm.nb_threads);
  return Coll_Alltoall_thread(sendbuf, sendcount, sendtype, 
                              recvbuf, recvcount, recvtype,
                              global_comm);
#else
  printf("Local: Thread %d, total_size %d, send_buf %p\n", global_comm.tid, global_comm.mpi_comm_size * global_comm.nb_threads, sendbuf);
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
  printf("MPI: Thread %d, total_size %d\n", global_comm.tid, global_comm.mpi_comm_size * global_comm.nb_threads);
  return Coll_Gather_thread(sendbuf, sendcount, sendtype, 
                            recvbuf, recvcount, recvtype,
                            root,
                            global_comm);
#else
  printf("Local: Thread %d, total_size %d, send_buf %p\n", global_comm.tid, global_comm.mpi_comm_size * global_comm.nb_threads, sendbuf);
  assert(0);
#endif  
}

int Coll_Allgather(void *sendbuf, int sendcount, collDataType_t sendtype, 
                   void *recvbuf, int recvcount, collDataType_t recvtype, 
                   Coll_Comm global_comm)
{
  #if defined(COLL_USE_MPI)
  printf("MPI: Thread %d, total_size %d\n", global_comm.tid, global_comm.mpi_comm_size * global_comm.nb_threads);
  return Coll_Allgather_thread(sendbuf, sendcount, sendtype, 
                               recvbuf, recvcount, recvtype,
                               global_comm);
#else
  printf("Local: Thread %d, total_size %d, send_buf %p\n", global_comm.tid, global_comm.mpi_comm_size * global_comm.nb_threads, sendbuf);
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
  printf("MPI: Thread %d, total_size %d\n", global_comm.tid, global_comm.mpi_comm_size * global_comm.nb_threads);
  return Coll_Bcast(buf, count, type, 
                    root,
                    global_comm);
#else
  printf("Local: Thread %d, total_size %d, send_buf %p\n", global_comm.tid, global_comm.mpi_comm_size * global_comm.nb_threads, buf);
  assert(0);
#endif 
}
