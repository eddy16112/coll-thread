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

int collAllgatherLocal(const void* sendbuf,
                       int sendcount,
                       collDataType_t sendtype,
                       void* recvbuf,
                       int recvcount,
                       collDataType_t recvtype,
                       collComm_t global_comm)
{
  assert(recvcount == sendcount);
  assert(sendtype == recvtype);

  int total_size = global_comm->global_comm_size;

  int sendtype_extent = get_dtype_size(sendtype);
  int recvtype_extent = get_dtype_size(recvtype);
  assert(sendtype_extent == recvtype_extent);

  int global_rank = global_comm->global_rank;

  void* sendbuf_tmp = NULL;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    sendbuf_tmp = (void*)malloc(sendtype_extent * sendcount);
    memcpy(sendbuf_tmp, recvbuf, sendtype_extent * sendcount);
    // int * sendval = (int*)sendbuf_tmp;
    // printf("malloc %p, size %ld, [%d]\n", sendbuf_tmp, total_size * recvtype_extent * recvcount,
    // sendval[0]);
  } else {
    sendbuf_tmp = const_cast<void*>(sendbuf);
  }

  global_comm->shared_data->buffers[global_rank] = const_cast<void*>(sendbuf_tmp);
  __sync_synchronize();

  int recvfrom_global_rank;
  for (int i = 0; i < total_size; i++) {
    recvfrom_global_rank = i;
    while (global_comm->shared_data->buffers[recvfrom_global_rank] == NULL)
      ;
    char* src = (char*)global_comm->shared_data->buffers[recvfrom_global_rank];
    char* dst = (char*)recvbuf + (ptrdiff_t)recvfrom_global_rank * recvtype_extent * recvcount;
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, dtype %d, copy rank %d (%p) to rank %d (%p)\n",
           i,
           global_rank,
           sendtype_extent,
           recvfrom_global_rank,
           src,
           global_rank,
           dst);
#endif
    memcpy(dst, src, sendcount * sendtype_extent);
  }

  collBarrierLocal(global_comm);
  if (sendbuf == recvbuf) { free(sendbuf_tmp); }

  __sync_synchronize();

  collUpdateBuffer(global_comm);

  return collSuccess;
}