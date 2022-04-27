
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>

#include "coll.h"

#define NTHREADS 8
#define NB_GROUPS 4
#define SEND_COUNT 800
#define COLL_DTYPE collInt
typedef int DTYPE;

#define VERIFICATION_2

typedef struct thread_args_s {
  int mpi_comm_size;
  int nb_threads;
  int mpi_rank;
  int tid;
  int group_id;
#if defined (LEGATE_USE_GASNET)
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
  
  sendbufs = (DTYPE **)malloc(sizeof(DTYPE*) * global_comm_size);
  recvbufs = (DTYPE **)malloc(sizeof(DTYPE*) * global_comm_size);
  sendcount = (int *)malloc(sizeof(int) * global_comm_size);
  recvcount = (int *)malloc(sizeof(int) * global_comm_size);
  sdispls = (int *)malloc(sizeof(int) * global_comm_size);
  rdispls = (int *)malloc(sizeof(int) * global_comm_size);

  #if defined (LEGATE_USE_GASNET)
  int *mapping_table = (int *)malloc(sizeof(int) * global_comm_size);
  for (int i = 0; i < global_comm_size; i++) {
    mapping_table[i] = i / args->nb_threads;
  }
  collCommCreate(&global_comm, global_comm_size, global_rank, args->group_id, mapping_table);
#else
  collCommCreate(&global_comm, global_comm_size, global_rank, args->group_id, NULL);
#endif

#if 0
  int total_size = 0;
  for (int i = 0; i < global_comm_size; i++) {
    total_size += (i+1);
  }
  sendbuf = (DTYPE *)malloc(sizeof(DTYPE) * total_size);
  for (int i = 0; i < total_size; i++) {
    sendbuf[i] = global_rank;
  }
  int soffset = 0;
  for (int i = 0; i < global_comm_size; i++) {
    sdispls[i] = soffset;
    soffset += i+1;
    sendcount[i] = i + 1;
  }

  if (global_rank == 0) {
    for (int i = 0; i < global_comm_size; i++) {
      printf("%d ", sdispls[i]);
    }
    printf("\n");
  }

  int seg_size = global_rank + 1;
  recvbuf = (DTYPE *)malloc(sizeof(DTYPE) * seg_size * global_comm_size);
  for (int i = 0; i < global_comm_size; i++) {
    recvcount[i] = seg_size;
    rdispls[i] = i * seg_size;
  }

  // test inplace
  for (int i = 0; i < seg_size * global_comm_size; i++) {
    recvbuf[i] = global_rank;
  }

  // Coll_Alltoallv(recvbuf, sendcount,
  //           sdispls, COLL_DTYPE,
  //           recvbuf, recvcount,
  //           rdispls, COLL_DTYPE, 
  //           &global_comm);

  collAlltoallv(sendbuf, sendcount,
                sdispls, COLL_DTYPE,
                recvbuf, recvcount,
                rdispls, COLL_DTYPE, 
                &global_comm);
  
  if (global_rank == 2) {
    for (int i = 0; i < seg_size * global_comm_size; i++) {
      printf("%d ", recvbuf[i]);
    }
    printf("\n");
  }
#else

  int seg_size = global_rank + 1;
  sendbuf = (DTYPE *)malloc(sizeof(DTYPE) * seg_size * global_comm_size);
  for (int i = 0; i < seg_size * global_comm_size; i++) {
    sendbuf[i] = global_rank;
  }

  for (int i = 0; i < global_comm_size; i++) {
    sendbufs[i] = sendbuf + i * seg_size;
    sendcount[i] = seg_size;
    sdispls[i] = i * seg_size;
  }

  collAllgather(&seg_size, 1, collInt, 
                recvcount, 1, collInt, 
                &global_comm);
  
  for (int i = 0; i < global_comm_size; i++) {
    assert(recvcount[i] == i + 1); 
    //recvcount[i] = i + 1;
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

  // for (int i = 0; i < 2048; i++) {
  //   recvbuf[i] = global_rank;
  // }

  if (global_rank == 0) {
    for (int i = 0; i < global_comm_size; i++) {
      printf("%d ", rdispls[i]);
    }
    printf("\n");
  }

  printf("global rank %d, recv total size %d , send size %d\n", global_rank, total_size, seg_size * global_comm_size);
  // for (int i = 0; i < seg_size * global_comm_size; i++) {
  //   printf("%d ", sendbuf[i]);
  // }
  // printf("\n");
  for (int i = 0; i < 10; i++) {
    collAlltoallv(sendbuf, sendcount,
                  sdispls, COLL_DTYPE,
                  recvbuf, recvcount,
                  rdispls, COLL_DTYPE, 
                  &global_comm);
  }
  // if (global_rank == 0) {
  //   for (int i = 0; i < total_size; i++) {
  //     printf("%d ", recvbuf[i]);
  //   }
  //   printf("\n");
  // }
  for (int i = 0; i < global_comm_size; i++) {
    DTYPE *tmp_recv = recvbufs[i];
    for (int j = 0; j < recvcount[i]; j++) {
      if (tmp_recv[j] != recvcount[i]-1) {
        printf("rank %d, i %d, j %d, recv %d, expect %d\n", global_rank, i, j, tmp_recv[j], recvcount[i]-1);
        assert(0);
      }
    }
  }
#endif

  collCommDestroy(&global_comm);

  return NULL;
}
 
int main( int argc, char *argv[] )
{
  int mpi_rank = 0;
  int global_rank = 0;
  int mpi_comm_size = 1;

  collInit(0, NULL);

#if defined (LEGATE_USE_GASNET)
  MPI_Comm  mpi_comm;  
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_comm_size);
#endif
 
 #if defined (LEGATE_USE_GASNET)
  MPI_Barrier(mpi_comm);
#endif

  pthread_t thread_id[NTHREADS*NB_GROUPS];
  thread_args_t args[NTHREADS*NB_GROUPS];

  int group_id[NB_GROUPS];
  for (int i = 0; i < NB_GROUPS; i++) {
    collGetUniqueId(&(group_id[i]));
    assert(group_id[i] == i);
  }

  for (int i = 0; i < NTHREADS*NB_GROUPS; i++) {
    args[i].mpi_rank = mpi_rank;
    args[i].mpi_comm_size = mpi_comm_size;
    args[i].tid = i % NTHREADS;
    args[i].nb_threads = NTHREADS;
    args[i].group_id = i / NTHREADS;
 #if defined (LEGATE_USE_GASNET)
    args[i].comm = mpi_comm;
  #endif
    pthread_create(&thread_id[i], NULL, thread_func, (void *)&(args[i]));
    //thread_func((void *)&(args[i]));
  }

  for(int i = 0; i < NTHREADS*NB_GROUPS; i++) {
      pthread_join( thread_id[i], NULL); 
  }

  collFinalize();
  return 0;
}