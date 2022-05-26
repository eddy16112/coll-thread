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

namespace legate {
namespace comm {
namespace coll {

int collAllgatherMPI(const void* sendbuf,
                     int sendcount,
                     CollDataType sendtype,
                     void* recvbuf,
                     int recvcount,
                     CollDataType recvtype,
                     CollComm global_comm)
{
  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_sendtype = collDtypeToMPIDtype(sendtype);

  MPI_Aint lb, sendtype_extent, recvtype_extent;
  MPI_Type_get_extent(mpi_sendtype, &lb, &sendtype_extent);

  void* sendbuf_tmp = const_cast<void*>(sendbuf);

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    sendbuf_tmp = collAllocateInplaceBuffer(recvbuf, sendtype_extent * sendcount);
  }

  collGatherMPI(sendbuf_tmp, sendcount, sendtype, recvbuf, recvcount, recvtype, 0, global_comm);

  collBcastMPI(recvbuf, recvcount * total_size, recvtype, 0, global_comm);

  if (sendbuf == recvbuf) { free(sendbuf_tmp); }

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate