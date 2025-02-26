#include <pmix.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <pthread.h>

#define BOOTSTRAP_PMIX_KEYSIZE 64

struct PMIX_Comm {
  pmix_proc_t proc;
  int mpi_rank;
  int mpi_comm_size;
  int tid;
  int nb_threads;
  int rank;
  int comm_size;
};

static pmix_proc_t myproc;

#define NTHREADS 1

struct thread_args_t {
  int mpi_comm_size;
  int nb_threads;
  int mpi_rank;
  int tid;
};

pthread_barrier_t barrier;


static pmix_status_t pmix_exchange(void)
{
  pmix_status_t status;
  pmix_info_t *info;
  bool flag = true;

  status = PMIx_Commit();
  assert(status == PMIX_SUCCESS);

  PMIX_INFO_CREATE(info, 2);
  PMIX_INFO_LOAD(&info[0], PMIX_COLLECT_DATA, &flag, PMIX_BOOL);
  PMIX_INFO_LOAD(&info[1], PMIX_THREADING_MODEL, "pthread", PMIX_STRING);

  status = PMIx_Fence(NULL, 0, info, 2);
  assert(status == PMIX_SUCCESS);

out:
  return status;
}

static pmix_status_t pmix_put(const char *key, const void *value,
    size_t valuelen)
{
  pmix_value_t val;
  pmix_status_t status;

  PMIX_VALUE_CONSTRUCT(&val);
  val.type = PMIX_BYTE_OBJECT;
  val.data.bo.bytes = (char *)value;
  val.data.bo.size = valuelen;

  status = PMIx_Put(PMIX_GLOBAL, key, &val);
  assert(status == PMIX_SUCCESS);

cleanup_val:
  val.data.bo.bytes = NULL;  // protect the data
  val.data.bo.size = 0;
  PMIX_VALUE_DESTRUCT(&val);

  return status;
}

static pmix_status_t pmix_get(PMIX_Comm &comm, int pe, const char *key, void *value,
    size_t valuelen)
{
  pmix_proc_t proc;
  pmix_value_t *val;
  pmix_status_t status;

  /* ensure the region is zero'd out */
  memset(value, 0, valuelen);

  /* setup the ID of the proc whose info we are getting */
  PMIX_LOAD_NSPACE(proc.nspace, comm.proc.nspace);

  proc.rank = (uint32_t)pe;

  status = PMIx_Get(&proc, key, NULL, 0, &val);
  assert(status == PMIX_SUCCESS);

  if (val == NULL) {
    goto out;
  }

  /* see if the data fits into the given region */
  if (valuelen < val->data.bo.size) {
    status = PMIX_ERROR;
    goto rel_val;
  }

  /* copy the results across */
  memcpy(value, val->data.bo.bytes, val->data.bo.size);

rel_val:
  PMIX_VALUE_RELEASE(val);
out:
  return status;
}

pmix_status_t pmix_allgather(const void *sendbuf, void *recvbuf, int length,
    PMIX_Comm &comm)
{
  pmix_status_t status;
  char key[BOOTSTRAP_PMIX_KEYSIZE];

  if (comm.comm_size == 1) {
    memcpy(recvbuf, sendbuf, length);
    return 0;
  }

  snprintf(key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-ALLGATHER-%04x", comm.tid);

  status = pmix_put(key, sendbuf, length);
  assert(status == PMIX_SUCCESS);

  pthread_barrier_wait(&barrier);
  status = pmix_exchange();
  assert(status == PMIX_SUCCESS);
  for (int i = 0; i < comm.mpi_comm_size; i++) {
    for (int j = 0; j < comm.nb_threads; j++) {
      char remote_key[BOOTSTRAP_PMIX_KEYSIZE];
      snprintf(remote_key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-ALLGATHER-%04x", j);
      int remote_rank = i * comm.nb_threads + j;
      //printf("Rank %d mpi_rank %d tid %d remote_rank %d\n", comm.rank, comm.mpi_rank, comm.tid, remote_rank);
      status = pmix_get(comm,i, remote_key, (char *)recvbuf + length * remote_rank, length);
      assert(status == PMIX_SUCCESS);
    }
  }

  return status;
}

void *thread_func(void *thread_args)
{
  thread_args_t *args = (thread_args_t*)thread_args;

  PMIX_Comm comm;
  comm.mpi_rank = args->mpi_rank;
  comm.mpi_comm_size = args->mpi_comm_size;
  comm.tid = args->tid;
  comm.nb_threads = args->nb_threads;
  comm.rank = comm.mpi_rank * comm.nb_threads + comm.tid;
  comm.comm_size = comm.mpi_comm_size * comm.nb_threads;

  const int N = 5;
  std::vector<int> sendbuf(N, comm.rank);

  std::vector<int> recvbuf(comm.comm_size * N);

  pmix_status_t status = pmix_allgather(sendbuf.data(), recvbuf.data(), N * sizeof(int), comm);
  assert(status == PMIX_SUCCESS);

  std::cout << "Rank " << comm.rank << " mpi_rank " << comm.mpi_rank << " received data: ";
  for (int i = 0; i < comm.comm_size; ++i) {
      std::cout << "[ ";
      for (int j = 0; j < N; ++j) {
          std::cout << recvbuf[i * N + j] << " ";
      }
      std::cout << "] ";
  }
  std::cout << std::endl;

  return NULL;
}

int main(int argc, char **argv) {
  
  printf("pid %d\n", getpid());
  // sleep(10);
  pmix_info_t *info;
  PMIX_INFO_CREATE(info, 1);
  PMIX_INFO_LOAD(&info[0], PMIX_THREADING_MODEL, "pthread", PMIX_STRING);

  PMIX_PROC_CONSTRUCT(&myproc);

  PMIx_Init(&myproc, info, 1);

  int mpi_rank = myproc.rank;

  pmix_proc_t proc;
  PMIX_LOAD_NSPACE(proc.nspace, proc.nspace);
  proc.rank = PMIX_RANK_WILDCARD;
  pmix_value_t *val;
  PMIx_Get(&proc, PMIX_JOB_SIZE, NULL, 0, &val);
  int mpi_comm_size = val->data.uint32;
  printf("Hello, world! I am rank %d of %d\n", mpi_rank, mpi_comm_size);


  pthread_t thread_id[NTHREADS];
  thread_args_t args[NTHREADS];

  pthread_barrier_init(&barrier, NULL, NTHREADS);

  for (int i = 0; i < NTHREADS; i++) {
    args[i].mpi_rank = mpi_rank;
    args[i].mpi_comm_size = mpi_comm_size;
    args[i].tid = i;
    args[i].nb_threads = NTHREADS;
    pthread_create(&thread_id[i], NULL, thread_func, (void *)&(args[i]));
    //thread_func((void *)&(args[i]));
  }

  for(int i = 0; i < NTHREADS; i++) {
      pthread_join( thread_id[i], NULL); 
  }
  pthread_barrier_destroy(&barrier);


  PMIx_Finalize(NULL, 0);
  return 0;
}