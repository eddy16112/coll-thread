
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
 #include <sys/time.h>

#include "coll.h"

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
  void *buf;
  int count; 
  CollDataType type;
  int root;
  int uid;
} thread_args_t;

void *thread_func(void *thread_args)
{
  thread_args_t *args = (thread_args_t*)thread_args;

  Coll_Comm global_comm;
  int global_rank = args->mpi_rank * args->nb_threads + args->tid;
  int global_comm_size = args->mpi_comm_size * args->nb_threads;

  int *mapping_table = (int *)malloc(sizeof(int) * global_comm_size);
  for (int i = 0; i < global_comm_size; i++) {
    mapping_table[i] = i / args->nb_threads;
  }
  collCommCreate(&global_comm, global_comm_size, global_rank, args->uid, mapping_table);

  bcastMPI(args->buf, args->count, args->type, 
            args->root, &global_comm);
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

#if defined (LEGATE_USE_GASNET)
  MPI_Comm  mpi_comm;  
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_comm_size);
#endif

  size_t N = mpi_comm_size * SEND_COUNT * NTHREADS;

  DTYPE **buffs;
  DTYPE *a;

  buffs = (DTYPE**)malloc(sizeof(DTYPE*) * NTHREADS);

  int root = 5;

  for (int i = 0; i < NTHREADS; i++) {
    a = (DTYPE *)malloc(N*sizeof(DTYPE));
 
    // printf("N %ld, rank=%d, tid %d, a=", N, mpi_rank, i);

    global_rank = mpi_rank * NTHREADS + i;
    for(int j = 0; j < N; j++)
    {
      if (global_rank == root) {
        a[j] = (DTYPE)j;
      } else {
        a[j] = (DTYPE)0;
      }
      // printf(" %d", a[j]);		
    }

    // printf("\n");
 
    buffs[i] = a;
  }
 
 #if defined (LEGATE_USE_GASNET)
  MPI_Barrier(mpi_comm);
#endif

  struct timeval tv;
  gettimeofday(&tv,NULL);
  unsigned long start_time = 1000000 * tv.tv_sec + tv.tv_usec;
  int uid = collInitComm();
  gettimeofday(&tv,NULL);
  unsigned long end_time = 1000000 * tv.tv_sec + tv.tv_usec;
  printf("time %ld\n", end_time-start_time);

  pthread_t thread_id[NTHREADS];
  thread_args_t args[NTHREADS];

  pthread_barrier_init(&barrier, NULL, NTHREADS);

  for (int i = 0; i < NTHREADS; i++) {
    args[i].mpi_rank = mpi_rank;
    args[i].mpi_comm_size = mpi_comm_size;
    args[i].tid = i;
    args[i].nb_threads = NTHREADS;
 #if defined (LEGATE_USE_GASNET)
    args[i].comm = mpi_comm;
  #endif
    args[i].buf = buffs[i];
    args[i].count = SEND_COUNT;
    args[i].type = COLL_DTYPE;
    args[i].root = root;
    args[i].uid = uid;
    pthread_create(&thread_id[i], NULL, thread_func, (void *)&(args[i]));
    //thread_func((void *)&(args[i]));
  }

  for(int i = 0; i < NTHREADS; i++) {
      pthread_join( thread_id[i], NULL); 
  }
  pthread_barrier_destroy(&barrier);

 #if defined (LEGATE_USE_GASNET)
	MPI_Barrier(mpi_comm);
#endif

  for (int i = 0; i < NTHREADS; i++) {
    a = buffs[i];
    global_rank = mpi_rank * NTHREADS + i;

    if (global_rank == root) {
      // printf("rank=%d, tid %d, b=", mpi_rank, i);
      // for(int j = 0; j < N; j++)
      // {
      //   printf(" %d", (int)a[j]);		
      // }
      // printf("\n");

      for (int x = 0; x < N; x ++) {
        if (a[x] != (DTYPE)x) {
          printf("x %d val %d\n", x, (int)a[x]);
          assert(0);
        }
      }

      printf("rank %d, tid %d, SUCCESS\n", mpi_rank, i);
    }

    free(a);
  }

  free(buffs);
 
  collFinalize();

#if defined (LEGATE_USE_GASNET)
  MPI_Finalize();
#endif

  return 0;
}