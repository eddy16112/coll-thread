
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>

#include "coll.h"

#define NTHREADS 16
#define SEND_COUNT 8
#define COLL_DTYPE collInt
typedef int DTYPE;

#define VERIFICATION_2

pthread_barrier_t barrier;

typedef struct thread_args_s {
  int mpi_comm_size;
  int nb_threads;
  int mpi_rank;
  int tid;
#if defined (COLL_USE_MPI)
  MPI_Comm comm;
#endif
  void *sendbuf;
  int sendcount; 
  collDataType_t sendtype;
  void *recvbuf;
  int recvcount;
  collDataType_t recvtype;
} thread_args_t;

void *thread_func(void *thread_args)
{
  thread_args_t *args = (thread_args_t*)thread_args;

  Coll_Comm global_comm;
  global_comm.mpi_comm_size = args->mpi_comm_size;
  global_comm.mpi_rank = args->mpi_rank;
  global_comm.nb_threads = args->nb_threads;
  global_comm.tid = args->tid;
  global_comm.global_rank = args->mpi_rank * args->nb_threads + args->tid;

 #if defined (COLL_USE_MPI)
  global_comm.comm = args->comm;
#endif

  Coll_Allgather(args->sendbuf, args->sendcount, args->sendtype, 
                 args->recvbuf, args->recvcount, args->recvtype,
                 global_comm);
  return NULL;
}
 
int main( int argc, char *argv[] )
{
  int mpi_rank = 0;
  int global_rank = 0;
  int mpi_comm_size = 1;

#if defined (COLL_USE_MPI) || defined (COLL_USE_NCCL)
  MPI_Comm  mpi_comm;  
  int provided;
 
  MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_comm_size);
  // pid_t pid = getpid();
  // printf("rank %d, pid %ld\n", mpi_rank, pid);
  // sleep(10);
#endif

  size_t N = mpi_comm_size * SEND_COUNT * NTHREADS;

  DTYPE **send_buffs, **recv_buffs;
  DTYPE *a, *b;

  send_buffs = (DTYPE**)malloc(sizeof(DTYPE*) * NTHREADS);
  recv_buffs = (DTYPE**)malloc(sizeof(DTYPE*) * NTHREADS);

  for (int i = 0; i < NTHREADS; i++) {
    a = (DTYPE *)malloc(SEND_COUNT*sizeof(DTYPE));
	  b = (DTYPE *)malloc(N*sizeof(DTYPE));
 
    printf("N %ld, rank=%d, tid %d, a=", N, mpi_rank, i);

    global_rank = mpi_rank * NTHREADS + i;
    for(int j = 0; j < SEND_COUNT; j++)
    {
      a[j] = (DTYPE)(global_rank * SEND_COUNT + j);
      b[j] = (DTYPE)0;
      // printf(" %d", a[j]);		
    }

    printf("\n");
 
    send_buffs[i] = a;
    recv_buffs[i] = b;
  }
 
 #if defined (COLL_USE_MPI) || defined (COLL_USE_NCCL)
  MPI_Barrier(mpi_comm);
#endif

  pthread_t thread_id[NTHREADS];
  thread_args_t args[NTHREADS];

  pthread_barrier_init(&barrier, NULL, NTHREADS);

  for (int i = 0; i < NTHREADS; i++) {
    args[i].mpi_rank = mpi_rank;
    args[i].mpi_comm_size = mpi_comm_size;
    args[i].tid = i;
    args[i].nb_threads = NTHREADS;
 #if defined (COLL_USE_MPI) || defined (COLL_USE_NCCL)
    args[i].comm = mpi_comm;
  #endif
    args[i].sendbuf = send_buffs[i];
    args[i].sendcount = SEND_COUNT;
    args[i].sendtype = COLL_DTYPE;
    args[i].recvbuf = recv_buffs[i];
    args[i].recvcount = SEND_COUNT;
    args[i].recvtype = COLL_DTYPE;
    pthread_create(&thread_id[i], NULL, thread_func, (void *)&(args[i]));
    //thread_func((void *)&(args[i]));
  }

  for(int i = 0; i < NTHREADS; i++) {
      pthread_join( thread_id[i], NULL); 
  }
  pthread_barrier_destroy(&barrier);

 #if defined (COLL_USE_MPI) || defined (COLL_USE_NCCL)
	MPI_Barrier(mpi_comm);
#endif

  for (int i = 0; i < NTHREADS; i++) {
    a = send_buffs[i];
    b = recv_buffs[i];
    global_rank = mpi_rank * NTHREADS + i;

    // printf("rank=%d, tid %d, b=", mpi_rank, i);
    // for(int j = 0; j < N; j++)
    // {
    //   printf(" %d", (int)b[j]);		
    // }
    // printf("\n");

    for (int x = 0; x < N; x ++) {
      if (b[x] != (DTYPE)x) {
        printf("x %d val %d\n", x, (int)b[x]);
        assert(0);
      }
    }

    printf("rank %d, tid %d, SUCCESS\n", mpi_rank, i);

    free(a);
    free(b);  
  }

  free(send_buffs);
  free(recv_buffs);
 
#if defined (COLL_USE_MPI) || defined (COLL_USE_NCCL)
  MPI_Finalize();
#endif
  return 0;
}