
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <string.h>

#define NTHREADS 16
#define SEND_COUNT 40
#define MPI_DTYPE MPI_INT
typedef int DTYPE;

#define ALLTOALL_USE_SENDRECV_OLD

#define DEBUG_PRINT

typedef struct thread_args_s {
  int mpi_comm_size;
  int nb_threads;
  int mpi_rank;
  int tid;
  MPI_Comm comm;
  void *sendbuf;
  int sendcount; 
  MPI_Datatype sendtype;
  void *recvbuf;
  int recvcount;
  MPI_Datatype recvtype;
} thread_args_t;

typedef struct MPI_Global_Comm_s {
  MPI_Comm comm;
  int mpi_comm_size;
  int nb_threads;
  int mpi_rank;
  int tid;
} MPI_Global_Comm;
 
int MPI_Alltoall_thread(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                        void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                        MPI_Global_Comm global_comm)
{	
  int base_tag = 0;
  int res;

  int total_size = global_comm.mpi_comm_size * global_comm.nb_threads;
	MPI_Status status;

  int sendtype_size;
  int recvtype_size;
  // MPI_Type_size(sendtype, &sendtype_size);
  // MPI_Type_size(recvtype, &recvtype_size);
  MPI_Aint lb, sendtype_extent, recvtype_extent;
  MPI_Type_get_extent(sendtype, &lb, &sendtype_extent);
  MPI_Type_get_extent(recvtype, &lb, &recvtype_extent);
  // assert(recvtype_extent == 4);
  // assert(sendtype_extent == 4);
 
  int global_rank = global_comm.mpi_rank * global_comm.nb_threads + global_comm.tid;

  void *sendbuf_tmp = NULL;
  if (sendbuf == MPI_IN_PLACE) {
    sendbuf_tmp = (void *)malloc(total_size * recvtype_extent * recvcount);
    memcpy(sendbuf_tmp, recvbuf, total_size * recvtype_extent * recvcount);
    int * sendval = (int*)sendbuf_tmp;
    printf("malloc %p, size %ld, [%d]\n", sendbuf_tmp, total_size * recvtype_extent * recvcount, sendval[0]);
  } else {
    sendbuf_tmp = sendbuf;
  }

#ifdef ALLTOALL_USE_SENDRECV
  int sendto_global_rank, recvfrom_global_rank, sendto_mpi_rank, recvfrom_mpi_rank;
	for(int i = 1 ; i < total_size + 1; i++) {
    sendto_global_rank  = (global_rank + i) % total_size;
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    char* src = (char*)sendbuf_tmp + (ptrdiff_t)sendto_global_rank * sendtype_extent * sendcount;
    char* dst = (char*)recvbuf + (ptrdiff_t)recvfrom_global_rank * recvtype_extent * recvcount;
    sendto_mpi_rank = sendto_global_rank / global_comm.nb_threads;
    recvfrom_mpi_rank = recvfrom_global_rank / global_comm.nb_threads;
    // tag: seg idx + rank_idx
    int send_tag = sendto_global_rank * 10000 + global_rank; // which dst seg it sends to (in dst rank)
    int recv_tag = global_rank * 10000 + recvfrom_global_rank; // idx of current seg we are receving (in src/my rank)
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, rank %d, tid %d, send %d to %d, send_tag %d, recv %d from %d, recv_tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, 
      sendto_global_rank, sendto_mpi_rank, send_tag, recvfrom_mpi_rank, recvfrom_mpi_rank, recv_tag);
#endif
    res = MPI_Sendrecv(src, sendcount, sendtype, sendto_mpi_rank, send_tag, dst, recvcount, recvtype, recvfrom_mpi_rank, recv_tag, global_comm.comm, &status);
    assert(res == MPI_SUCCESS);
	}
#elif defined (ALLTOALL_USE_SENDRECV_OLD)
    int dest_mpi_rank;
	for(int i = 0 ; i < total_size; i++) {
    char* src = (char*)sendbuf_tmp + i * sendtype_extent * sendcount;
    char* dst = (char*)recvbuf + i * recvtype_extent * recvcount;
    dest_mpi_rank = i / global_comm.nb_threads;
    int send_tag = i * 10000 + global_rank; // which seg it sends to
    int recv_tag = global_rank * 10000 + i; // idx of current seg
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, rank %d, tid %d, send %d to %d, send_tag %d, recv %d from %d, recv_tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, i, dest_mpi_rank, send_tag, i, dest_mpi_rank, recv_tag);
#endif
    res = MPI_Sendrecv(src, sendcount, sendtype, dest_mpi_rank, send_tag, dst, recvcount, recvtype, dest_mpi_rank, recv_tag, global_comm.comm, &status);
    assert(res == MPI_SUCCESS);
	}
#else
  int tag;
  int dest_mpi_rank;
	for(int i = 0 ; i < total_size; i++) {
    char* src = (char*)sendbuf_tmp + i * sendtype_extent * sendcount;
    dest_mpi_rank = i / global_comm.nb_threads;
    tag = i * 10000 + global_rank;
    res = MPI_Send(src, sendcount, sendtype, dest_mpi_rank, tag, global_comm.comm);
    assert(res == MPI_SUCCESS);
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, rank %d, tid %d, send %d to %d, send_tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, i, dest_mpi_rank, tag);
#endif
	}
  
  for(int i = 0 ; i < total_size; i++) {
    char* dst = (char*)recvbuf + i * recvtype_extent * recvcount;
    int dest_mpi_rank = i / global_comm.nb_threads;
    tag = global_rank * 10000 + i;
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, rank %d, tid %d, recv %d from %d, recv_tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, i, dest_mpi_rank, tag);
#endif
    res = MPI_Recv(dst, recvcount, recvtype, dest_mpi_rank, tag, global_comm.comm, &status);
    assert(res == MPI_SUCCESS);
	}
#endif

  return 0;
}

void *thread_func(void *thread_args)
{
  thread_args_t *args = (thread_args_t*)thread_args;

  int total_size = args->mpi_comm_size * args->nb_threads;
  printf("Thread %d, total_size %d\n", args->tid, total_size);

  MPI_Global_Comm global_comm;
  global_comm.mpi_comm_size = args->mpi_comm_size;
  global_comm.mpi_rank = args->mpi_rank;
  global_comm.comm = args->comm;
  global_comm.nb_threads = args->nb_threads;
  global_comm.tid = args->tid;

  MPI_Alltoall_thread(args->sendbuf, args->sendcount, args->sendtype, 
                      args->recvbuf, args->recvcount, args->recvtype,
                      global_comm);
}
 
int main( int argc, char *argv[] )
{
  int mpi_rank;//iam
  int mpi_comm_size;//np
  MPI_Comm  mpi_comm;  
  int provided;
 
  MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_comm_size);

  size_t N = mpi_comm_size * SEND_COUNT * NTHREADS;

  DTYPE **send_buffs, **recv_buffs;
  DTYPE *a, *b;

  send_buffs = (DTYPE**)malloc(sizeof(DTYPE*) * NTHREADS);
  recv_buffs = (DTYPE**)malloc(sizeof(DTYPE*) * NTHREADS);

  for (int i = 0; i < NTHREADS; i++) {
    a = (DTYPE *)malloc(N*sizeof(DTYPE));
	  b = (DTYPE *)malloc(N*sizeof(DTYPE));
 
    printf("N %ld, rank=%d, tid %d, a=", N, mpi_rank, i);
    for(int i = 0; i < N; i++)
    {
      a[i] = (DTYPE)i;
      b[i] = (DTYPE)i;
      //printf(" %d", a[i]);		
    }		
    printf("\n");
 
    send_buffs[i] = a;
    recv_buffs[i] = b;
  }
 
  MPI_Barrier(mpi_comm);

  pthread_t thread_id[NTHREADS];
  thread_args_t args[NTHREADS];

  for (int i = 0; i < NTHREADS; i++) {
    args[i].mpi_rank = mpi_rank;
    args[i].mpi_comm_size = mpi_comm_size;
    args[i].tid = i;
    args[i].nb_threads = NTHREADS;
    args[i].comm = mpi_comm;
    args[i].sendbuf = send_buffs[i];
    args[i].sendbuf = MPI_IN_PLACE;
    //args[i].sendcount = SEND_COUNT;
    args[i].sendtype = MPI_INT;
    args[i].recvbuf = recv_buffs[i];
    args[i].recvcount = SEND_COUNT;
    args[i].recvtype = MPI_INT;
    pthread_create(&thread_id[i], NULL, thread_func, (void *)&(args[i]));
  }

  for(int i = 0; i < NTHREADS; i++) {
      pthread_join( thread_id[i], NULL); 
  }
	
	MPI_Barrier(mpi_comm);

  int global_rank;
  for (int i = 0; i < NTHREADS; i++) {
    a = send_buffs[i];
    b = recv_buffs[i];
    global_rank = mpi_rank * NTHREADS + i;
 
    // printf("rank=%d, tid %d, b=", mpi_rank, i);
    // for(int i = 0; i < N; i++)
    // {
    //   printf(" %d", (int)b[i]);		
    // }
    // printf("\n");
    
    int start_value = global_rank * SEND_COUNT;
    for (int i = 0; i < N; i += SEND_COUNT) {
      for (int j = 0; j < SEND_COUNT; j++) {
        if (b[i+j] != (DTYPE)start_value + j) {
          printf("i %d j %d, val %d\n", i, j, (int)b[i+j]);
          assert(0);
        }
      }
    }

    printf("rank %d, tid %d, SUCCESS\n", mpi_rank, i);

    free(a);
    free(b);  
  }

  free(send_buffs);
  free(recv_buffs);
 
  MPI_Finalize();
  return 0;
}