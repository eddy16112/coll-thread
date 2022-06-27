
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>

#include "realm.h"
#include "coll.h"

using namespace Realm;

using namespace legate::comm::coll;

#define NTHREADS 16
#define SEND_COUNT 80
#define COLL_DTYPE CollDataType::CollInt
typedef int DTYPE;

#define VERIFICATION_2

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
  collCommCreate(&global_comm, global_comm_size, global_rank, args->uid, mapping_table);
#else
  collCommCreate(&global_comm, global_comm_size, global_rank, args->uid, NULL);
#endif
  assert(global_comm.mpi_comm_size == global_comm.mpi_comm_size_actual);

#if 0
    // Define the buffer containing the values to send
    int* buffer_send;
    int buffer_send_length;
    switch(global_rank)
    {
        case 0:
            buffer_send_length = 3;
            buffer_send = (int*)malloc(sizeof(int) * buffer_send_length);
            buffer_send[0] = 0;
            buffer_send[1] = 100;
            buffer_send[2] = 200;
            printf("Process %d, my values = %d, %d, %d.\n", global_rank, buffer_send[0], buffer_send[1], buffer_send[2]);
            break;
        case 1:
            buffer_send_length = 3;
            buffer_send = (int*)malloc(sizeof(int) * buffer_send_length);
            buffer_send[0] = 300;
            buffer_send[1] = 400;
            buffer_send[2] = 500;
            printf("Process %d, my values = %d, %d, %d.\n", global_rank, buffer_send[0], buffer_send[1], buffer_send[2]);
            break;
        case 2:
            buffer_send_length = 1;
            buffer_send = (int*)malloc(sizeof(int) * buffer_send_length);
            buffer_send[0] = 600;
            printf("Process %d, my value = %d.\n", global_rank, buffer_send[0]);
            break;
    }
 
    // Define my counts for sending (how many integers do I send to each process?)
    int counts_send[3];
    switch(global_rank)
    {
        case 0:
            counts_send[0] = 1;
            counts_send[1] = 2;
            counts_send[2] = 0;
            break;
        case 1:
            counts_send[0] = 0;
            counts_send[1] = 0;
            counts_send[2] = 3;
            break;
        case 2:
            counts_send[0] = 1;
            counts_send[1] = 0;
            counts_send[2] = 0;
            break;
    }
 
    // Define my displacements for sending (where is located in the buffer each message to send?)
    int displacements_send[3];
    switch(global_rank)
    {
        case 0:
            displacements_send[0] = 0;
            displacements_send[1] = 1;
            displacements_send[2] = 0;
            break;
        case 1:
            displacements_send[0] = 0;
            displacements_send[1] = 0;
            displacements_send[2] = 0;
            break;
        case 2:
            displacements_send[0] = 0;
            displacements_send[1] = 0;
            displacements_send[2] = 0;
            break;
    }
 
    // Define the buffer for reception
    int* buffer_recv;
    int buffer_recv_length;
    switch(global_rank)
    {
        case 0:
            buffer_recv_length = 2;
            buffer_recv = (int*)malloc(sizeof(int) * buffer_recv_length);
            break;
        case 1:
            buffer_recv_length = 2;
            buffer_recv = (int*)malloc(sizeof(int) * buffer_recv_length);
            break;
        case 2:
            buffer_recv_length = 3;
            buffer_recv = (int*)malloc(sizeof(int) * buffer_recv_length);
            break;
    }
 
    // Define my counts for receiving (how many integers do I receive from each process?)
    int counts_recv[3];
    switch(global_rank)
    {
        case 0:
            counts_recv[0] = 1;
            counts_recv[1] = 0;
            counts_recv[2] = 1;
            break;
        case 1:
            counts_recv[0] = 2;
            counts_recv[1] = 0;
            counts_recv[2] = 0;
            break;
        case 2:
            counts_recv[0] = 0;
            counts_recv[1] = 3;
            counts_recv[2] = 0;
            break;
    }
 
    // Define my displacements for reception (where to store in buffer each message received?)
    int displacements_recv[3];
    switch(global_rank)
    {
        case 0:
            displacements_recv[0] = 1;
            displacements_recv[1] = 0;
            displacements_recv[2] = 0;
            break;
        case 1:
            displacements_recv[0] = 0;
            displacements_recv[1] = 0;
            displacements_recv[2] = 0;
            break;
        case 2:
            displacements_recv[0] = 0;
            displacements_recv[1] = 0;
            displacements_recv[2] = 0;
            break;
    }
 
    collAlltoallv(buffer_send, counts_send, displacements_send, 
                  buffer_recv, counts_recv, displacements_recv, CollDataType::CollInt, &global_comm);
    
    printf("Values received on process %d:", global_rank);
    for(int i = 0; i < buffer_recv_length; i++)
    {
        printf(" %d", buffer_recv[i]);
    }
    printf("\n");
 
    free(buffer_send);
    free(buffer_recv);
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
    for (int i = 0; i< 5; i++) {
        collAlltoallv( sbuf, sendcounts, sdispls,
                        rbuf, recvcounts, rdispls, CollDataType::CollInt, &global_comm );
        // if (global_comm.global_rank%2 == 0) sleep(3);
        printf("rank %d, iter %d\n", global_rank, i);
        /* Check rbuf */
        for (int i=0; i<global_comm_size; i++) {
            p = rbuf + rdispls[i];
            for (int j=0; j<global_rank; j++) {
                // if (global_rank == 2) printf("%d ", p[j]);
                if (p[j] != i * 100 + (global_rank*(global_rank+1))/2 + j) {
                    fprintf( stderr, "[%d] got %d expected %d for %dth\n",
                                        global_rank, p[j],(i*(i+1))/2 + j, j );
                    fflush(stderr);
                    err++;
                }
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

  collInit(0, NULL);

  Runtime rt;

  rt.init(&argc, &argv);

#if defined (LEGATE_USE_GASNET)
  MPI_Comm  mpi_comm;  
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_comm_size);
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
    pthread_create(&thread_id[i], NULL, thread_func, (void *)&(args[i]));
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