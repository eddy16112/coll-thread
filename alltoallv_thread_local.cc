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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coll.h"
#include "legion.h"

namespace legate {
namespace comm {
namespace coll {

using namespace Legion;
extern Logger log_coll;

int collAlltoallvLocal(const void* sendbuf,
                       const int sendcounts[],
                       const int sdispls[],
                       CollDataType sendtype,
                       void* recvbuf,
                       const int recvcounts[],
                       const int rdispls[],
                       CollDataType recvtype,
                       CollComm global_comm)
{
  int res;

  assert(sendtype == recvtype);

  int total_size = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int sendtype_extent = collGetDtypeSize(sendtype);
  int recvtype_extent = collGetDtypeSize(recvtype);

  void* sendbuf_tmp = NULL;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    int total_send_count = sdispls[total_size - 1] + sendcounts[total_size - 1];
    sendbuf_tmp          = (void*)malloc(sendtype_extent * total_send_count);
    assert(sendbuf_tmp != NULL);
    memcpy(sendbuf_tmp, recvbuf, sendtype_extent * total_send_count);
  } else {
    sendbuf_tmp = const_cast<void*>(sendbuf);
  }

  global_comm->comm->displs[global_rank]  = sdispls;
  global_comm->comm->buffers[global_rank] = sendbuf_tmp;
  __sync_synchronize();

  int recvfrom_global_rank;
  int recvfrom_seg_id = global_rank;
  const void* src_base      = nullptr;
  const int* displs         = nullptr;
  for (int i = 1; i < total_size + 1; i++) {
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    while (global_comm->comm->buffers[recvfrom_global_rank] == nullptr ||
           global_comm->comm->displs[recvfrom_global_rank] == nullptr)
      ;
    src_base  = global_comm->comm->buffers[recvfrom_global_rank];
    displs    = global_comm->comm->displs[recvfrom_global_rank];
    char* src = static_cast<char*>(const_cast<void*>(src_base)) + static_cast<ptrdiff_t>(displs[recvfrom_seg_id]) * sendtype_extent;
    char* dst = static_cast<char*>(recvbuf) + static_cast<ptrdiff_t>(rdispls[recvfrom_global_rank]) * recvtype_extent;
#ifdef DEBUG_PRINT
    log_coll.debug(
      "i: %d === global_rank %d, dtype %d, copy rank %d (seg %d, sdispls %d, %p) to rank %d (seg "
      "%d, rdispls %d, %p)",
      i,
      global_rank,
      sendtype_extent,
      recvfrom_global_rank,
      recvfrom_seg_id,
      sdispls[recvfrom_seg_id],
      src,
      global_rank,
      recvfrom_global_rank,
      rdispls[recvfrom_global_rank],
      dst);
#endif
    memcpy(dst, src, recvcounts[recvfrom_global_rank] * recvtype_extent);
  }

  collBarrierLocal(global_comm);
  if (sendbuf == recvbuf) { free(sendbuf_tmp); }

  __sync_synchronize();

  collUpdateBuffer(global_comm);
  collBarrierLocal(global_comm);

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate