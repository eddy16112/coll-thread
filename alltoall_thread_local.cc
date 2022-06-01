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

int alltoallLocal(const void* sendbuf,
                      void* recvbuf,
                      int count,
                      CollDataType type,
                      CollComm global_comm)
{
  int res;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int type_extent = collGetDtypeSize(type);

  const void* sendbuf_tmp = sendbuf;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    sendbuf_tmp = collAllocateInplaceBuffer(recvbuf, total_size * type_extent * count);
  }

  global_comm->comm->buffers[global_rank] = sendbuf_tmp;
  __sync_synchronize();

  int recvfrom_global_rank;
  int recvfrom_seg_id  = global_rank;
  const void* src_base = nullptr;
  for (int i = 1; i < total_size + 1; i++) {
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    // wait for other threads to update the buffer address
    while (global_comm->comm->buffers[recvfrom_global_rank] == nullptr)
      ;
    src_base  = global_comm->comm->buffers[recvfrom_global_rank];
    char* src = static_cast<char*>(const_cast<void*>(src_base)) +
                static_cast<ptrdiff_t>(recvfrom_seg_id) * type_extent * count;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "i: %d === global_rank %d, dtype %d, copy rank %d (seg %d, %p) to rank %d (seg %d, %p)",
      i,
      global_rank,
      sendtype_extent,
      recvfrom_global_rank,
      recvfrom_seg_id,
      src,
      global_rank,
      recvfrom_global_rank,
      dst);
#endif
    memcpy(dst, src, count * type_extent);
  }

  collBarrierLocal(global_comm);
  if (sendbuf == recvbuf) { free(const_cast<void*>(sendbuf_tmp)); }

  __sync_synchronize();

  collUpdateBuffer(global_comm);
  collBarrierLocal(global_comm);

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate