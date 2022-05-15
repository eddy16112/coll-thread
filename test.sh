set -e
mpirun -np 16 ./alltoall_test
mpirun -np 16 ./alltoallv_test
mpirun -np 16 ./myalltoallv_test
mpirun -np 16 ./alltoallv_inplace_test
mpirun -np 16 ./alltoallv_con_test
mpirun -np 16 ./alltoallv_con_test2
mpirun -np 16 ./allgather_test
