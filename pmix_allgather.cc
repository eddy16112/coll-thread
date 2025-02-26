#include <pmix.h>
#include <assert.h>
#include <iostream>
#include <vector>

#define BOOTSTRAP_PMIX_KEYSIZE 64

struct PMIX_Comm {
  pmix_proc_t proc;
  int rank;
  int size;
  int key_index;
};


static pmix_status_t pmix_exchange(void)
{
  pmix_status_t status;
  pmix_info_t info;
  bool flag = true;

  status = PMIx_Commit();
  assert(status == PMIX_SUCCESS);

  PMIX_INFO_CONSTRUCT(&info);
  PMIX_INFO_LOAD(&info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);

  status = PMIx_Fence(NULL, 0, &info, 1);
  assert(status == PMIX_SUCCESS);

destruct_info:
  PMIX_INFO_DESTRUCT(&info);
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

  if (comm.size == 1) {
    memcpy(recvbuf, sendbuf, length);
    return 0;
  }

  snprintf(key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-ALLGATHER-%04x", comm.key_index);

  status = pmix_put(key, sendbuf, length);
  assert(status == PMIX_SUCCESS);

  status = pmix_exchange();
  assert(status == PMIX_SUCCESS);
  for (int i = 0; i < comm.size; i++) {
    // assumes that same length is passed by all the processes
    status = pmix_get(comm,i, key, (char *)recvbuf + length * i, length);
    assert(status == PMIX_SUCCESS);
  }

  return status;
}

int main(int argc, char **argv) {
  PMIX_Comm comm;
  PMIX_PROC_CONSTRUCT(&(comm.proc));

  PMIx_Init(&(comm.proc), NULL, 0);

  comm.rank = comm.proc.rank;

  pmix_proc_t proc;
  PMIX_LOAD_NSPACE(proc.nspace, proc.nspace);
  proc.rank = PMIX_RANK_WILDCARD;
  pmix_value_t *val;
  PMIx_Get(&proc, PMIX_JOB_SIZE, NULL, 0, &val);
  comm.size = val->data.uint32;
  comm.key_index = 1;
  printf("Hello, world! I am rank %d of %d\n", comm.rank, comm.size);

  // 定义 sendbuf 为一个长度为 N 的数组，内容为 rank
    const int N = 5; // 假设 N 为 5
    std::vector<int> sendbuf(N, comm.rank);

    // 接收缓冲区，大小为 comm.size * N
    std::vector<int> recvbuf(comm.size * N);

    // 调用 pmix_allgather 函数
    pmix_status_t status = pmix_allgather(sendbuf.data(), recvbuf.data(), N * sizeof(int), comm);
    assert(status == PMIX_SUCCESS);

    std::cout << "Rank " << comm.rank << " received data: ";
    for (int i = 0; i < comm.size; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < N; ++j) {
            std::cout << recvbuf[i * N + j] << " ";
        }
        std::cout << "] ";
    }
    std::cout << std::endl;


  PMIx_Finalize(NULL, 0);
  return 0;
}