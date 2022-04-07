/* Copyright 2022 Stanford University
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
 */


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "../coll.h"
#include "legion.h"
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  ALLTOALL_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

typedef struct task_args_s{
  int nb_threads;
  int sendcount;
} task_args_t;

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int nb_threads = 1;
  int sendcount = 8;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-t"))
        nb_threads = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-c"))
        sendcount = atoi(command_args.argv[++i]);
    }
  }
  task_args_t task_arg;
  task_arg.nb_threads = nb_threads;
  task_arg.sendcount = sendcount;
#ifdef COLL_USE_MPI
  int mpi_rank, mpi_comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
  int num_subregions = nb_threads * mpi_comm_size;
  printf("running top level task on %d node, %d total threads\n", mpi_comm_size, num_subregions);
#else
  int num_subregions = nb_threads;
  printf("running top level task on single node, %d threads\n", num_subregions);
#endif

  int num_elements = num_subregions * sendcount * num_subregions; 

  // Create our logical regions using the same schemas as earlier examples
  Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect); 
  runtime->attach_name(is, "is");
  FieldSpace input_fs = runtime->create_field_space(ctx);
  runtime->attach_name(input_fs, "input_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(int),FID_X);
    runtime->attach_name(input_fs, FID_X, "X");
  }
  FieldSpace output_fs = runtime->create_field_space(ctx);
  runtime->attach_name(output_fs, "output_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(int),FID_Z);
    runtime->attach_name(output_fs, FID_Z, "Z");
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  runtime->attach_name(input_lr, "input_lr");
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);
  runtime->attach_name(output_lr, "output_lr");

  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  runtime->attach_name(ip, "ip");

  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp, "input_lp");
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);
  runtime->attach_name(output_lp, "output_lp");

  // Create our launch domain.  Note that is the same as color domain
  // as we are going to launch one task for each subregion we created.
  ArgumentMap arg_map;

  {
    IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
                                TaskArgument(&task_arg, sizeof(task_args_t)), arg_map);
    init_launcher.add_region_requirement(
        RegionRequirement(input_lp, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr));
    init_launcher.region_requirements[0].add_field(FID_X);
    runtime->execute_index_space(ctx, init_launcher);
  }

  {
    IndexLauncher daxpy_launcher(ALLTOALL_TASK_ID, color_is,
                                TaskArgument(&task_arg, sizeof(task_args_t)), arg_map);
    daxpy_launcher.add_region_requirement(
        RegionRequirement(input_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, input_lr));
    daxpy_launcher.region_requirements[0].add_field(FID_X);
    daxpy_launcher.add_region_requirement(
        RegionRequirement(output_lp, 0/*projection ID*/,
                          WRITE_DISCARD, EXCLUSIVE, output_lr));
    daxpy_launcher.region_requirements[1].add_field(FID_Z);
    runtime->execute_index_space(ctx, daxpy_launcher);
  }
                    
  IndexLauncher check_launcher(CHECK_TASK_ID, color_is,
                               TaskArgument(&task_arg, sizeof(task_args_t)), arg_map);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[0].add_field(FID_Z);
  runtime->execute_index_space(ctx, check_launcher);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, is);
  runtime->destroy_index_space(ctx, color_is);
}

void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  const task_args_t task_arg = *((const task_args_t*)task->args);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];

  const FieldAccessor<WRITE_DISCARD,int,1,coord_t,
        Realm::AffineAccessor<int,1,coord_t> > acc(regions[0], fid);
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  int* ptr = acc.ptr(rect.lo);
  //printf("Initializing field %d for block %d, buf %p, pid " IDFMT "\n", fid, point, ptr, task->current_proc.id);

  int mpi_rank = 0;
#if defined (COLL_USE_MPI)
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
  int tid = point % task_arg.nb_threads;
  // int global_rank = mpi_rank * task_arg.nb_threads + tid;
  int global_rank = point;

  assert(rect.volume() > task_arg.sendcount);
  for (int i = 0; i < task_arg.sendcount; i++) {
    ptr[i] = global_rank * task_arg.sendcount + i;
  }
}

void alltoall_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const int point = task->index_point.point_data[0];
  const task_args_t task_arg = *((const task_args_t*)task->args);

  const FieldAccessor<READ_ONLY,int,1,coord_t,
          Realm::AffineAccessor<int,1,coord_t> > sendacc(regions[0], FID_X);
  const FieldAccessor<WRITE_DISCARD,int,1,coord_t,
          Realm::AffineAccessor<int,1,coord_t> > recvacc(regions[1], FID_Z);

  printf("Running coll for point %d pid " IDFMT "\n", 
          point, task->current_proc.id);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  const int *sendbuf = sendacc.ptr(rect.lo);
  int *recvbuf = recvacc.ptr(rect.lo);

  // printf("send %p, recv %p\n", sendbuf, recvbuf);

  Coll_Comm global_comm;

#if defined (COLL_USE_MPI)
  int mpi_rank, mpi_comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
  global_comm.mpi_comm_size = mpi_comm_size;
  global_comm.mpi_rank = mpi_rank;
  global_comm.comm = MPI_COMM_WORLD;
#else
  global_comm.mpi_comm_size = 1;
  global_comm.mpi_rank = 0;
#endif
  global_comm.nb_threads = task_arg.nb_threads;
  global_comm.tid = point % task_arg.nb_threads;
  global_comm.global_rank = point;

  Coll_Allgather((void*)sendbuf, task_arg.sendcount, collInt, 
                 (void*)recvbuf, task_arg.sendcount, collInt,
                 global_comm);
  printf("Point %d, Allgather Done\n", point);

}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  const int point = task->index_point.point_data[0];
  const task_args_t task_arg = *((const task_args_t*)task->args);

  const FieldAccessor<READ_ONLY,int,1,coord_t,
        Realm::AffineAccessor<int,1,coord_t> > recvacc(regions[0], FID_Z);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());

  const int *recvbuf = recvacc.ptr(rect.lo);

  for (int i = 0; i < (int)rect.volume(); i++) {
    if (recvbuf[i] != i) {
      printf("point %d, i %d, val %d, expect %d\n", 
        point, i, recvbuf[i], i);
      assert(0);
    }
  }
  printf("Point %d SUCCESS!\n", point);
  // bool all_passed = true;
  // for (PointInRectIterator<1> pir(rect); pir(); pir++)
  // {
  //   int received = acc_z[*pir];
  //   printf("%d ", received);
  //   // if (expected != received)
  //   //   all_passed = false;
  // }
  // printf("\n");
  // if (all_passed)
  //   printf("SUCCESS!\n");
  // else
  //   printf("FAILURE!\n");
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

#ifdef COLL_USE_MPI
  int provided;
 
  MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE, &provided);
#endif

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }

  {
    TaskVariantRegistrar registrar(ALLTOALL_TASK_ID, "alltoall");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<alltoall_task>(registrar, "alltoall");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  int val = Runtime::start(argc, argv);
#ifdef COLL_USE_MPI
  MPI_Finalize();
#endif
  return val;
}
