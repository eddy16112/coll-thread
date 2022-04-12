
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
} thread_args_t;

void *thread_func(void *thread_args)
{
  thread_args_t *args = (thread_args_t*)thread_args;

  Coll_Comm global_comm;
  int global_rank = args->mpi_rank * args->nb_threads + args->tid;
  int global_comm_size = args->mpi_comm_size * args->nb_threads;

  DTYPE *sendbuf, *recvbuf;
  DTYPE **sendbufs, **recvbufs;
  int *sendcount, *recvcount;
  int *sdispls, *rdispls;

  int seg_size = global_rank + 1;
  sendbuf = (DTYPE *)malloc(sizeof(DTYPE) * seg_size * global_comm_size);
  for (int i = 0; i < seg_size * global_comm_size; i++) {
    sendbuf[i] = global_rank;
  }
  
  sendbufs = (DTYPE **)malloc(sizeof(DTYPE*) * global_comm_size);
  recvbufs = (DTYPE **)malloc(sizeof(DTYPE*) * global_comm_size);
  sendcount = (int *)malloc(sizeof(int) * global_comm_size);
  recvcount = (int *)malloc(sizeof(int) * global_comm_size);
  sdispls = (int *)malloc(sizeof(int) * global_comm_size);
  rdispls = (int *)malloc(sizeof(int) * global_comm_size);

  for (int i = 0; i < global_comm_size; i++) {
    sendbufs[i] = sendbuf + i * seg_size;
    sendcount[i] = seg_size;
    sdispls[i] = i * seg_size;
  }

#if defined (COLL_USE_MPI)
  int *mapping_table = (int *)malloc(sizeof(int) * global_comm_size);
  for (int i = 0; i < global_comm_size; i++) {
    mapping_table[i] = i / args->nb_threads;
  }
  Coll_Create_comm(&global_comm, global_comm_size, global_rank, mapping_table);
#else
  Coll_Create_comm(&global_comm, global_comm_size, global_rank, NULL);
#endif

  Coll_Allgather(&seg_size, 1, collInt, 
                 recvcount, 1, collInt, 
                 &global_comm);
  
  for (int i = 0; i < global_comm_size; i++) {
    assert(recvcount[i] == i + 1); 
    // recvcount[i] = i + 1;
  }

  int total_size = 0;
  for (int i = 0; i < global_comm_size; i++) {
    total_size += (i+1);
  }
  recvbuf = (DTYPE *)malloc(sizeof(DTYPE) * total_size);
  DTYPE *tmpbuf = recvbuf;
  int tmp_seg_size;
  int roffset = 0;
  for (int i = 0; i < global_comm_size; i++) {
    recvbufs[i] = tmpbuf;
    rdispls[i] = roffset;
    tmp_seg_size = i+1;
    roffset += i+1;
    tmpbuf += tmp_seg_size;
  }

  if (global_rank == 0) {
    for (int i = 0; i < global_comm_size; i++) {
      printf("%d ", rdispls[i]);
    }
    printf("\n");
  }

  printf("global rank, recv total size %d \n", total_size);
  // for (int i = 0; i < seg_size * global_comm_size; i++) {
  //   printf("%d ", sendbuf[i]);
  // }
  // printf("\n");
  
  Coll_Alltoallv(sendbuf, sendcount,
                 sdispls, COLL_DTYPE,
                 recvbuf, recvcount,
                 rdispls, COLL_DTYPE, 
                 &global_comm);
  // if (global_rank == 2) {
  //   for (int i = 0; i < total_size; i++) {
  //     printf("%d ", recvbuf[i]);
  //   }
  //   printf("\n");
  // }
  for (int i = 0; i < global_comm_size; i++) {
    DTYPE *tmp_recv = recvbufs[i];
    for (int j = 0; j < recvcount[i]; j++) {
      if (tmp_recv[j] != recvcount[i]-1) {
        printf("i %d, j %d, recv %d, expect %d\n", i, j, tmp_recv[j], recvcount[i]-1);
        assert(0);
      }
    }
  }
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
    pthread_create(&thread_id[i], NULL, thread_func, (void *)&(args[i]));
    //thread_func((void *)&(args[i]));
  }

  for(int i = 0; i < NTHREADS; i++) {
      pthread_join( thread_id[i], NULL); 
  }
  pthread_barrier_destroy(&barrier);

 
#if defined (COLL_USE_MPI) || defined (COLL_USE_NCCL)
  MPI_Finalize();
#endif
  return 0;
}