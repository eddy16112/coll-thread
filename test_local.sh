set -e
mpirun -np 1 ./alltoall_test
mpirun -np 1 ./alltoallv_test
mpirun -np 1 ./myalltoallv_test
mpirun -np 1 ./alltoallv_inplace_test
mpirun -np 1 ./alltoallv_con_test
mpirun -np 1 ./alltoallv_con_test2
mpirun -np 1 ./allgather_test