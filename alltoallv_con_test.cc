
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>

#include "coll.h"

#define NTHREADS 4
#define NB_GROUPS 8
#define SEND_COUNT 800
#define COLL_DTYPE CollInt
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

  #if defined (LEGATE_USE_GASNET)
  int *mapping_table = (int *)malloc(sizeof(int) * global_comm_size);
  for (int i = 0; i < global_comm_size; i++) {
    mapping_table[i] = i / args->nb_threads;
  }
  collCommCreate(&global_comm, global_comm_size, global_rank, args->group_id, mapping_table);
#else
  collCommCreate(&global_comm, global_comm_size, global_rank, args->group_id, NULL);
#endif

  if (global_comm.unique_id %2 == 0) {
    //sleep(5);
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

    collAllgather(&seg_size, 1, CollInt, 
                  recvcount, 1, CollInt, 
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

    // if (global_rank == 0) {
    //   for (int i = 0; i < global_comm_size; i++) {
    //     printf("%d ", rdispls[i]);
    //   }
    //   printf("\n");
    // }

    printf("global rank %d, recv total size %d , send size %d\n", global_rank, total_size, seg_size * global_comm_size);
    // for (int i = 0; i < seg_size * global_comm_size; i++) {
    //   printf("%d ", sendbuf[i]);
    // }
    // printf("\n");
    for (int i = 0; i < 1; i++) {
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
  } else {
#if 1
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

  // if (global_rank == 1) {
  //   for (int i = 0; i < seg_size * global_comm_size; i++) {
  //     printf("%d ", recvbuf[i]);
  //   }
  //   printf("\n");
  // }
#else
    int *sbuf, *rbuf;
    int *sendcounts, *recvcounts, *rdispls, *sdispls;
    int *p, err;
    sbuf = (int *)malloc( global_comm_size * global_comm_size * sizeof(int) );
    rbuf = (int *)malloc( global_comm_size * global_comm_size * sizeof(int) );
    if (!sbuf || !rbuf) {
        fprintf( stderr, "Could not allocated buffers!\n" );
    }
    /* Load up the buffers */
    for (int i=0; i<global_comm_size*global_comm_size; i++) {
        sbuf[i] = i + 100*global_rank;
        rbuf[i] = -i;
    }
    /* Create and load the arguments to alltoallv */
    sendcounts = (int *)malloc( global_comm_size * sizeof(int) );
    recvcounts = (int *)malloc( global_comm_size * sizeof(int) );
    rdispls = (int *)malloc( global_comm_size * sizeof(int) );
    sdispls = (int *)malloc( global_comm_size * sizeof(int) );
    if (!sendcounts || !recvcounts || !rdispls || !sdispls) {
        fprintf( stderr, "Could not allocate arg items!\n" );fflush(stderr);
    }
    for (int i=0; i<global_comm_size; i++) {
        sendcounts[i] = i;
        recvcounts[i] = global_rank;
        rdispls[i] = i * global_rank;
        sdispls[i] = (i * (i+1))/2;
    }
    collAlltoallv( sbuf, sendcounts, sdispls, collInt,
                       rbuf, recvcounts, rdispls, collInt, &global_comm );
    /* Check rbuf */
    for (int i=0; i<global_comm_size; i++) {
        p = rbuf + rdispls[i];
        for (int j=0; j<global_rank; j++) {
            // printf("%d ", p[j]);
            if (p[j] != i * 100 + (global_rank*(global_rank+1))/2 + j) {
                fprintf( stderr, "[%d] got %d expected %d for %dth\n",
                                    global_rank, p[j],(i*(i+1))/2 + j, j );
                fflush(stderr);
                err++;
            }
        }
    }
    // printf("\n");
    free( sdispls );
    free( rdispls );
    free( recvcounts );
    free( sendcounts );
    free( rbuf );
    free( sbuf );
  #endif
  }

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
    //assert(group_id[i] == i);
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