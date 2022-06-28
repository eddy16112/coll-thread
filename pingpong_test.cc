
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>

#include "realm.h"
#include "coll.h"

using namespace Realm;

using namespace legate::comm::coll;

#define NTHREADS 4
#define SEND_COUNT 8
#define COLL_DTYPE CollDataType::CollInt
typedef int DTYPE;

#define VERIFICATION_2

pthread_barrier_t barrier;

typedef struct thread_args_s {
  int mpi_comm_size;
  int nb_threads;
  int mpi_rank;
  int tid;
#if defined (LEGATE_USE_GASNET)
  MPI_Comm comm;
#endif
  int uid;
} thread_args_t;

void *pingpong_func(void *thread_args)
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
  collCommCreate(&global_comm, global_comm_size, global_rank, args->uid, mapping_table);
#else
  collCommCreate(&global_comm, global_comm_size, global_rank, args->uid, NULL);
#endif

  int p2pbuf = 0;
  for (int i = 0; i < 10; i++) {
    if (global_rank % 2 == 0) {
      collSend(&p2pbuf, 1, CollDataType::CollInt, global_rank+1, global_rank, &global_comm);
      collRecv(&p2pbuf, 1, CollDataType::CollInt, global_rank+1, global_rank+1, &global_comm);
    } else {
      collRecv(&p2pbuf, 1, CollDataType::CollInt, global_rank-1, global_rank-1, &global_comm);
      p2pbuf ++;
      collSend(&p2pbuf, 1, CollDataType::CollInt, global_rank-1, global_rank, &global_comm);
    }
  }
  printf("global rank %d, buffer %d\n", global_rank, p2pbuf);
  assert(p2pbuf == 10);
  collCommDestroy(&global_comm);
  return NULL;
}

void *pingping_func(void *thread_args)
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
  collCommCreate(&global_comm, global_comm_size, global_rank, args->uid, mapping_table);
#else
  collCommCreate(&global_comm, global_comm_size, global_rank, args->uid, NULL);
#endif

  int p2pbuf[10];
  for (int i = 0; i < 10; i++) {
    if (global_rank % 2 == 0) {
      p2pbuf[i] = i;
      collSend(&(p2pbuf[i]), 1, CollDataType::CollInt, global_rank+1, global_rank, &global_comm);
    } else {
      collRecv(&(p2pbuf[i]), 1, CollDataType::CollInt, global_rank-1, global_rank-1, &global_comm);
    }
  }
  
  for (int i = 0; i < 10; i++) {
    if (global_rank % 2 != 0) {
      assert(p2pbuf[i] == i);
    }
  }
  collCommDestroy(&global_comm);
  return NULL;
}
 
int main( int argc, char *argv[] )
{
  int mpi_rank = 0;
  int global_rank = 0;
  int mpi_comm_size = 1;

#if defined (LEGATE_USE_GASNET)
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#endif

  collInit(argc, argv);

  Runtime rt;

  rt.init(&argc, &argv);

#if defined (LEGATE_USE_GASNET)
  MPI_Comm  mpi_comm;  
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_comm_size);
  // pid_t pid = getpid();
  // printf("rank %d, pid %ld\n", mpi_rank, pid);
  // sleep(10);
#endif
 
 #if defined (LEGATE_USE_GASNET)
  MPI_Barrier(mpi_comm);
#endif
  int uid = collInitComm();

  pthread_t thread_id[NTHREADS];
  thread_args_t args[NTHREADS];

  for (int i = 0; i < NTHREADS; i++) {
    args[i].mpi_rank = mpi_rank;
    args[i].mpi_comm_size = mpi_comm_size;
    args[i].tid = i;
    args[i].nb_threads = NTHREADS;
 #if defined (LEGATE_USE_GASNET)
    args[i].comm = mpi_comm;
  #endif
    args[i].uid = uid;
    pthread_create(&thread_id[i], NULL, pingpong_func, (void *)&(args[i]));
    //thread_func((void *)&(args[i]));
  }

  for(int i = 0; i < NTHREADS; i++) {
      pthread_join( thread_id[i], NULL); 
  }

#if defined (LEGATE_USE_GASNET)
	MPI_Barrier(mpi_comm);
#endif

  uid = collInitComm();

for (int i = 0; i < NTHREADS; i++) {
    args[i].mpi_rank = mpi_rank;
    args[i].mpi_comm_size = mpi_comm_size;
    args[i].tid = i;
    args[i].nb_threads = NTHREADS;
 #if defined (LEGATE_USE_GASNET)
    args[i].comm = mpi_comm;
  #endif
    args[i].uid = uid;
    pthread_create(&thread_id[i], NULL, pingping_func, (void *)&(args[i]));
    //thread_func((void *)&(args[i]));
  }

  for(int i = 0; i < NTHREADS; i++) {
      pthread_join( thread_id[i], NULL); 
  }

  collFinalize();

#if defined (LEGATE_USE_GASNET)
  MPI_Finalize();
#endif

  return 0;
}