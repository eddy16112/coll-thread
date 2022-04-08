
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>

#include "coll.h"

#define NTHREADS 16
#define SEND_COUNT 80
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
  int global_rank = args->mpi_rank * args->nb_threads + args->tid;
  int global_comm_size = args->mpi_comm_size * args->nb_threads;

 #if defined (COLL_USE_MPI)
  int *mapping_table = (int *)malloc(sizeof(int) * global_comm_size);
  for (int i = 0; i < global_comm_size; i++) {
    mapping_table[i] = i / args->nb_threads;
  }
  Coll_Create_comm(&global_comm, global_comm_size, global_rank, mapping_table);
#else
  Coll_Create_comm(&global_comm, global_comm_size, global_rank, NULL);
#endif

  Coll_Alltoall(args->sendbuf, args->sendcount, args->sendtype, 
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
#endif

  size_t N = mpi_comm_size * SEND_COUNT * NTHREADS;

  DTYPE **send_buffs, **recv_buffs;
  DTYPE *a, *b;

  send_buffs = (DTYPE**)malloc(sizeof(DTYPE*) * NTHREADS);
  recv_buffs = (DTYPE**)malloc(sizeof(DTYPE*) * NTHREADS);

  for (int i = 0; i < NTHREADS; i++) {
    a = (DTYPE *)malloc(N*sizeof(DTYPE));
	  b = (DTYPE *)malloc(N*sizeof(DTYPE));
 
    printf("N %ld, rank=%d, tid %d, a=", N, mpi_rank, i);
#ifdef VERIFICATION_1    
    for(int j = 0; j < N; j++)
    {
      a[j] = (DTYPE)j;
      b[j] = (DTYPE)j;
      //printf(" %d", a[i]);		
    }
#else
    global_rank = mpi_rank * NTHREADS + i;
    for(int j = 0; j < N; j++)
    {
      a[j] = (DTYPE)(global_rank * N + j);
      b[j] = (DTYPE)(global_rank * N + j);
      // printf(" %d", a[j]);		
    }
#endif		
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
    //args[i].sendbuf = recv_buffs[i];
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

#ifdef VERIFICATION_1    
    int start_value = global_rank * SEND_COUNT;
    for (int x = 0; x < N; x += SEND_COUNT) {
      for (int y = 0; y < SEND_COUNT; y++) {
        if (b[x+y] != (DTYPE)start_value + y) {
          printf("x %d y %d, val %d\n", x, y, (int)b[x+y]);
          assert(0);
        }
      }
    }
#else
    int ct_x = 0;
    for (int x = 0; x < N; x += SEND_COUNT) {
      int start_value = global_rank * SEND_COUNT + ct_x * N;
      for (int y = 0; y < SEND_COUNT; y++) {
        if (b[x+y] != (DTYPE)start_value + y) {
          printf("x %d y %d, val %d\n", x, y, (int)b[x+y]);
          assert(0);
        }
      }
      ct_x ++;
    }
#endif

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