#include <stdio.h>

#include "coll.h"

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
