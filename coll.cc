/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <atomic>
#include <cstdlib>

#include "coll.h"

#if defined(LEGATE_USE_GASNET)
MPI_Datatype collChar   = MPI_CHAR;
MPI_Datatype collInt8   = MPI_INT8_T;
MPI_Datatype collUint8  = MPI_UINT8_T;
MPI_Datatype collInt    = MPI_INT;
MPI_Datatype collUint32 = MPI_UINT32_T;
MPI_Datatype collInt64  = MPI_INT64_T;
MPI_Datatype collUint64 = MPI_UINT64_T;
MPI_Datatype collFloat  = MPI_FLOAT;
MPI_Datatype collDouble = MPI_DOUBLE;
#else
#include <stdint.h>

volatile shared_data_t* shared_data[MAX_NB_COMMS];

static bool coll_local_inited = false;

size_t get_dtype_size(collDataType_t dtype)
{
  if (dtype == collInt8 || dtype == collChar) {
    return sizeof(char);
  } else if (dtype == collUint8) {
    return sizeof(uint8_t);
  } else if (dtype == collInt) {
    return sizeof(int);
  } else if (dtype == collUint32) {
    return sizeof(uint32_t);
  } else if (dtype == collInt64) {
    return sizeof(int64_t);
  } else if (dtype == collUint64) {
    return sizeof(uint64_t);
  } else if (dtype == collFloat) {
    return sizeof(float);
  } else if (dtype == collDouble) {
    return sizeof(double);
  } else {
    assert(0);
    return -1;
  }
}
#endif

static std::atomic<int> current_unique_id(0);

int collCommCreate(collComm_t global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table)
{
  global_comm->global_comm_size = global_comm_size;
  global_comm->global_rank      = global_rank;
  global_comm->status           = true;
  global_comm->unique_id        = unique_id;
#if defined(LEGATE_USE_GASNET)
  int mpi_rank, mpi_comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
  global_comm->mpi_comm_size = mpi_comm_size;
  global_comm->mpi_rank      = mpi_rank;
  global_comm->comm          = MPI_COMM_WORLD;
  if (mapping_table != NULL) {
    global_comm->mapping_table.global_rank = (int*)malloc(sizeof(int) * global_comm_size);
    global_comm->mapping_table.mpi_rank    = (int*)malloc(sizeof(int) * global_comm_size);
    memcpy(global_comm->mapping_table.mpi_rank, mapping_table, sizeof(int) * global_comm_size);
    for (int i = 0; i < global_comm_size; i++) { global_comm->mapping_table.global_rank[i] = i; }
  }
#else
  global_comm->mpi_comm_size = 1;
  global_comm->mpi_rank      = 0;
  if (global_comm->global_rank == 0) {
    shared_data_t* data = (shared_data_t*)malloc(sizeof(shared_data_t));
    for (int j = 0; j < MAX_NB_THREADS; j++) {
      data->buffers[j] = NULL;
      data->displs[j]  = NULL;
    }
    data->barrier = (pthread_barrier_t*)malloc(sizeof(pthread_barrier_t));
    pthread_barrier_init(data->barrier, NULL, global_comm->global_comm_size);
    data->ready_flag                    = true;
    shared_data[global_comm->unique_id] = data;
  }
  __sync_synchronize();
  volatile shared_data_t* data = shared_data[global_comm->unique_id];
  while (data == NULL) { data = shared_data[global_comm->unique_id]; }
  global_comm->shared_data = shared_data[global_comm->unique_id];
  assert(global_comm->shared_data->ready_flag == true);
  printf("comm created rank %d\n", global_comm->global_rank);
#endif
  if (global_comm->global_comm_size % global_comm->mpi_comm_size == 0) {
    global_comm->nb_threads = global_comm->global_comm_size / global_comm->mpi_comm_size;
  } else {
    global_comm->nb_threads = global_comm->global_comm_size / global_comm->mpi_comm_size + 1;
  }
  return collSuccess;
}

int collCommDestroy(collComm_t global_comm)
{
#if defined(LEGATE_USE_GASNET)
  if (global_comm->mapping_table.global_rank != NULL) {
    free(global_comm->mapping_table.global_rank);
    global_comm->mapping_table.global_rank = NULL;
  }
  if (global_comm->mapping_table.mpi_rank != NULL) {
    free(global_comm->mapping_table.mpi_rank);
    global_comm->mapping_table.mpi_rank = NULL;
  }
#else
  if (global_comm->global_rank == 0) {
    shared_data_t* data = (shared_data_t*)shared_data[global_comm->unique_id];
    pthread_barrier_destroy(data->barrier);
    free(data->barrier);
    data->ready_flag = false;
    free(data);
    shared_data[global_comm->unique_id] = NULL;
  }
  __sync_synchronize();
  volatile shared_data_t* data = shared_data[global_comm->unique_id];
  while (data != NULL) { data = shared_data[global_comm->unique_id]; }
  printf("comm destroy rank %d\n", global_comm->global_rank);
#endif
  global_comm->status = false;
  return collSuccess;
}

int collAlltoallv(const void* sendbuf,
                  const int sendcounts[],
                  const int sdispls[],
                  collDataType_t sendtype,
                  void* recvbuf,
                  const int recvcounts[],
                  const int rdispls[],
                  collDataType_t recvtype,
                  collComm_t global_comm)
{
  printf("Alltoallv: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d\n",
         global_comm->global_rank,
         global_comm->mpi_rank,
         global_comm->unique_id,
         global_comm->global_comm_size);
#if defined(LEGATE_USE_GASNET)
  return collAlltoallvMPI(
    sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, global_comm);
#else
  return collAlltoallvLocal(
    sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, global_comm);
#endif
}

int collAlltoall(const void* sendbuf,
                 int sendcount,
                 collDataType_t sendtype,
                 void* recvbuf,
                 int recvcount,
                 collDataType_t recvtype,
                 collComm_t global_comm)
{
  printf("Alltoall: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d\n",
         global_comm->global_rank,
         global_comm->mpi_rank,
         global_comm->unique_id,
         global_comm->global_comm_size);
#if defined(LEGATE_USE_GASNET)
  return collAlltoallMPI(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, global_comm);
#else
  return collAlltoallLocal(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, global_comm);
#endif
}

int collGather(const void* sendbuf,
               int sendcount,
               collDataType_t sendtype,
               void* recvbuf,
               int recvcount,
               collDataType_t recvtype,
               int root,
               collComm_t global_comm)
{
#if defined(LEGATE_USE_GASNET)
  return collGatherMPI(
    sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, global_comm);
#else
  printf("Not implemented\n");
  assert(0);
#endif
}

int collAllgather(const void* sendbuf,
                  int sendcount,
                  collDataType_t sendtype,
                  void* recvbuf,
                  int recvcount,
                  collDataType_t recvtype,
                  collComm_t global_comm)
{
  printf("Allgather: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d\n",
         global_comm->global_rank,
         global_comm->mpi_rank,
         global_comm->unique_id,
         global_comm->global_comm_size);
#if defined(LEGATE_USE_GASNET)
  return collAllgatherMPI(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, global_comm);
#else
  return collAllgatherLocal(
    sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, global_comm);
#endif
}

int collBcast(void* buf, int count, collDataType_t type, int root, collComm_t global_comm)
{
  printf("Bcast: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d\n",
         global_comm->global_rank,
         global_comm->mpi_rank,
         global_comm->unique_id,
         global_comm->global_comm_size);
#if defined(LEGATE_USE_GASNET)
  return collBcast(buf, count, type, root, global_comm);
#else
  printf("Not implemented\n");
  assert(0);
#endif
}

// called from main thread
int collInit(int argc, char* argv[])
{
  current_unique_id = 0;
#if defined(LEGATE_USE_GASNET)
  int provided;
  return MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#else
  for (int i = 0; i < MAX_NB_COMMS; i++) { shared_data[i] = NULL; }

  coll_local_inited = true;
  return collSuccess;
#endif
}

int collFinalize(void)
{
#if defined(LEGATE_USE_GASNET)
  return MPI_Finalize();
#else
  assert(coll_local_inited == true);
  for (int i = 0; i < MAX_NB_COMMS; i++) { assert(shared_data[i] == NULL); }
  coll_local_inited = false;
  return collSuccess;
#endif
}

int collGetUniqueId(int* id)
{
  *id = current_unique_id;
  current_unique_id++;
#if defined(LEGATE_USE_GASNET)
  current_unique_id = current_unique_id % 10;
#else
  assert(current_unique_id <= MAX_NB_COMMS);
#endif
  return collSuccess;
}

#ifdef LEGATE_USE_GASNET
int collGenerateAlltoallTag(int rank1, int rank2, collComm_t global_comm)
{
  // tag: seg idx + rank_idx + tag
  // int send_tag = ((sendto_global_rank * 10000 + global_rank) * 10 + ALLTOALL_TAG) * 10 +
  // global_comm->unique_id; // which dst seg it sends to (in dst rank) int recv_tag = ((global_rank
  // * 10000 + recvfrom_global_rank) * 10 + ALLTOALL_TAG) * 10 + global_comm->unique_id; // idx of
  // current seg we are receving (in src/my rank)
#if 1
  int tag = ((rank1 * 10000 + rank2) * MAX_COLL_TYPES + ALLTOALL_TAG) * 10 + global_comm->unique_id;
#else
  int tag =
    ((rank1 % global_comm->nb_threads * 10000 + rank2) * MAX_COLL_TYPES + ALLTOALL_TAG) * 10 +
    global_comm->unique_id;
#endif
  assert(tag < INT_MAX && tag > 0);
  return tag;
}

int collGenerateAlltoallvTag(int rank1, int rank2, collComm_t global_comm)
{
  // tag: seg idx + rank_idx + tag
  // int send_tag = ((sendto_global_rank * 10000 + global_rank) * 10 + ALLTOALLV_TAG) * 10 +
  // global_comm->unique_id; // which dst seg it sends to (in dst rank) int recv_tag = ((global_rank
  // * 10000 + recvfrom_global_rank) * 10 + ALLTOALLV_TAG) * 10 + global_comm->unique_id; // idx of
  // current seg we are receving (in src/my rank)
#if 1
  int tag =
    ((rank1 * 10000 + rank2) * MAX_COLL_TYPES + ALLTOALLV_TAG) * 10 + global_comm->unique_id;
#else
  int tag =
    ((rank1 % global_comm->nb_threads * 10000 + rank2) * MAX_COLL_TYPES + ALLTOALLV_TAG) * 10 +
    global_comm->unique_id;
#endif
  assert(tag < INT_MAX && tag > 0);
  return tag;
}

int collGenerateBcastTag(int rank, collComm_t global_comm)
{
  int tag = (rank * MAX_COLL_TYPES + BCAST_TAG) * 10 + global_comm->unique_id;
  assert(tag < INT_MAX && tag >= 0);
  return tag;
}

int collGenerateGatherTag(int rank, collComm_t global_comm)
{
  int tag = (rank * MAX_COLL_TYPES + GATHER_TAG) * 10 + global_comm->unique_id;
  assert(tag < INT_MAX && tag > 0);
  return tag;
}

#else
void collUpdateBuffer(collComm_t global_comm)
{
  int global_rank                           = global_comm->global_rank;
  global_comm->shared_data->buffers[global_rank]       = NULL;
  global_comm->shared_data->displs[global_rank]        = NULL;
  // printf("rank %d, buffer idx %d\n", global_comm->global_rank, global_comm->current_buffer_idx);
}

void collBarrierLocal(collComm_t global_comm)
{
  assert(coll_local_inited == true);
  pthread_barrier_wait(global_comm->shared_data->barrier);
}
#endif