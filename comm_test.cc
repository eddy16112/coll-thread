
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>

#include "realm.h"
#include "coll.h"

using namespace Realm;

using namespace legate::comm::coll;
Logger log_coll("coll");

#define MAX_NB_COMMS 28

static std::vector<MPI_Comm> mpi_comms;
 
int main( int argc, char *argv[] )
{
  int mpi_rank = 0;
  int global_rank = 0;
  int mpi_comm_size = 1;
  int provided, res;
  res = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  mpi_comms.resize(MAX_NB_COMMS, MPI_COMM_NULL);
  for (int i = 0; i < MAX_NB_COMMS; i++) {
    res = MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comms[i]);
    assert(res == MPI_SUCCESS);
  }

  Runtime rt;

  rt.init(&argc, &argv);

  MPI_Comm  mpi_comm;  
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_comm_size);
  
  for (int i = 0; i < MAX_NB_COMMS; i++) {
    res = MPI_Comm_free(&mpi_comms[i]);
    assert(res == MPI_SUCCESS);
  }
  mpi_comms.clear();
  MPI_Finalize();
  
  return 0;
}