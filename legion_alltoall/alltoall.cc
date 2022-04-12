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
#include <unistd.h>
#include "../coll.h"
#include "legion.h"
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  ALLTOALL_TASK_ID,
  CHECK_TASK_ID,
  INIT_MAPPING_TASK_ID,
  INIT_COMM_CPU_TASK_ID,
  FINALIZE_COMM_CPU_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Z,
  FID_RANK,
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
  int sendcount = 80;
  int missing_nodes = 0;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-t"))
        nb_threads = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-c"))
        sendcount = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-m"))
        missing_nodes = atoi(command_args.argv[++i]);
    }
  }
  task_args_t task_arg;
  task_arg.nb_threads = nb_threads;
  task_arg.sendcount = sendcount;
#ifdef COLL_USE_MPI
  int mpi_rank, mpi_comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
  int num_subregions = nb_threads * (mpi_comm_size-missing_nodes);
  printf("running top level task on %d node, %d total threads, sendcount %d\n", mpi_comm_size, num_subregions, sendcount);
#else
  int num_subregions = nb_threads;
  printf("running top level task on single node, %d threads, sendcount %d\n", num_subregions, sendcount);
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
  FieldSpace mapping_fs = runtime->create_field_space(ctx);
  runtime->attach_name(mapping_fs, "mapping_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, mapping_fs);
    allocator.allocate_field(sizeof(int),FID_RANK);
    runtime->attach_name(mapping_fs, FID_RANK, "RANK");
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  runtime->attach_name(input_lr, "input_lr");
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);
  runtime->attach_name(output_lr, "output_lr");

  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  LogicalRegion mapping_lr = runtime->create_logical_region(ctx, color_is, mapping_fs);
  runtime->attach_name(mapping_lr, "mapping_lr");

  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  runtime->attach_name(ip, "ip");

  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp, "input_lp");
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);
  runtime->attach_name(output_lp, "output_lp");

  IndexPartition mapping_ip = runtime->create_equal_partition(ctx, color_is, color_is);
  runtime->attach_name(mapping_ip, "mapping_ip");
  LogicalPartition mapping_lp = runtime->get_logical_partition(ctx, mapping_lr, mapping_ip);
  runtime->attach_name(mapping_lp, "mapping_lp");

  // Create our launch domain.  Note that is the same as color domain
  // as we are going to launch one task for each subregion we created.
  ArgumentMap arg_map;
  {
    IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
                                TaskArgument(NULL, 0), arg_map);
    init_launcher.add_region_requirement(
        RegionRequirement(input_lp, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr));
    init_launcher.region_requirements[0].add_field(FID_X);
    runtime->execute_index_space(ctx, init_launcher);
  }
  {
    IndexLauncher init_mapping_launcher(INIT_MAPPING_TASK_ID, color_is, 
                                        TaskArgument(NULL, 0), arg_map);
    init_mapping_launcher.add_region_requirement(
        RegionRequirement(mapping_lp, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, mapping_lr));
    init_mapping_launcher.region_requirements[0].add_field(FID_RANK);
    runtime->execute_index_space(ctx, init_mapping_launcher);
  }

  IndexLauncher init_comm_cpu_launcher(INIT_COMM_CPU_TASK_ID, color_is, 
                                      TaskArgument(NULL, 0), arg_map);
  init_comm_cpu_launcher.add_region_requirement(
      RegionRequirement(mapping_lr, 0/*projection ID*/, 
                        READ_ONLY, EXCLUSIVE, mapping_lr));
  init_comm_cpu_launcher.region_requirements[0].add_field(FID_RANK);
  FutureMap comm_future_map = runtime->execute_index_space(ctx, init_comm_cpu_launcher);

  {
    IndexLauncher alltoall_launcher(ALLTOALL_TASK_ID, color_is,
                                    TaskArgument(&task_arg, sizeof(task_args_t)), arg_map);
    alltoall_launcher.point_futures.push_back(ArgumentMap(comm_future_map));
    alltoall_launcher.add_region_requirement(
        RegionRequirement(input_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, input_lr));
    alltoall_launcher.region_requirements[0].add_field(FID_X);
    alltoall_launcher.add_region_requirement(
        RegionRequirement(output_lp, 0/*projection ID*/,
                          WRITE_DISCARD, EXCLUSIVE, output_lr));
    alltoall_launcher.region_requirements[1].add_field(FID_Z);
    runtime->execute_index_space(ctx, alltoall_launcher);
  }
  // {
  //   IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
  //                               TaskArgument(NULL, 0), arg_map);
  //   init_launcher.add_region_requirement(
  //       RegionRequirement(output_lp, 0/*projection ID*/, 
  //                         WRITE_DISCARD, EXCLUSIVE, output_lr));
  //   init_launcher.region_requirements[0].add_field(FID_Z);
  //   runtime->execute_index_space(ctx, init_launcher);
  // }
  // {
  //   IndexLauncher alltoall_launcher(ALLTOALL_TASK_ID, color_is,
  //                                   TaskArgument(&task_arg, sizeof(task_args_t)), arg_map);
  //   alltoall_launcher.point_futures.push_back(ArgumentMap(comm_future_map));
  //   alltoall_launcher.add_region_requirement(
  //       RegionRequirement(input_lp, 0/*projection ID*/,
  //                         READ_ONLY, EXCLUSIVE, input_lr));
  //   alltoall_launcher.region_requirements[0].add_field(FID_X);
  //   alltoall_launcher.add_region_requirement(
  //       RegionRequirement(output_lp, 0/*projection ID*/,
  //                         WRITE_DISCARD, EXCLUSIVE, output_lr));
  //   alltoall_launcher.region_requirements[1].add_field(FID_Z);
  //   runtime->execute_index_space(ctx, alltoall_launcher);
  // }
                    
  IndexLauncher check_launcher(CHECK_TASK_ID, color_is,
                               TaskArgument(&task_arg, sizeof(task_args_t)), arg_map);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[0].add_field(FID_Z);
  runtime->execute_index_space(ctx, check_launcher);

  runtime->issue_execution_fence(ctx);
  {
    IndexLauncher finalize_comm_cpu_launcher(FINALIZE_COMM_CPU_TASK_ID, color_is, 
                                             TaskArgument(NULL, 0), arg_map);
    finalize_comm_cpu_launcher.point_futures.push_back(ArgumentMap(comm_future_map));
    runtime->execute_index_space(ctx, finalize_comm_cpu_launcher);
  }


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
  //printf("Initializing field %d for block %d, size %ld, pid " IDFMT "\n", fid, point, rect.volume(), task->current_proc.id);

  for (size_t i = 0; i < rect.volume(); i++) {
    ptr[i] = i;
  }
}

void init_mapping_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

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

  assert(rect.volume() == 1);

  int mpi_rank = 0;
#if defined (COLL_USE_MPI)
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
  ptr[0] = mpi_rank;

  // for (size_t i = 0; i < rect.volume(); i++) {
  //   ptr[i] = i;
  // }
}

Coll_Comm* init_comm_cpu_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  const int point = task->index_point.point_data[0];

  const FieldAccessor<READ_ONLY,int,1,coord_t,
          Realm::AffineAccessor<int,1,coord_t> > mappingacc(regions[0], FID_RANK);

  Rect<1> rect_mapping = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  const int *mapping_ptr = mappingacc.ptr(rect_mapping.lo);

  printf("Init Comm for point %d pid " IDFMT ", index size %ld, mapping size %ld\n", 
        point, task->current_proc.id, task->index_domain.get_volume(), rect_mapping.volume());

  // for (size_t i = 0; i < rect_mapping.volume(); i++) {
  //   printf("%d ", mapping_ptr[i]);
  // }
  // printf("\n");

  Coll_Comm *global_comm = (Coll_Comm*)malloc(sizeof(Coll_Comm));
  int global_rank = point;
  int global_comm_size = task->index_domain.get_volume();

 #if defined (COLL_USE_MPI)
  Coll_Create_comm(global_comm, global_comm_size, global_rank, mapping_ptr);
#else
  Coll_Create_comm(global_comm, global_comm_size, global_rank, NULL);
#endif

  assert(mapping_ptr[point] == global_comm->mpi_rank);
  assert(global_comm_size == rect_mapping.volume());
  return global_comm;
}

void alltoall_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const int point = task->index_point.point_data[0];
  const task_args_t task_arg = *((const task_args_t*)task->args);
  Coll_Comm *global_comm = task->futures[0].get_result<Coll_Comm*>();

  const FieldAccessor<READ_ONLY,int,1,coord_t,
          Realm::AffineAccessor<int,1,coord_t> > sendacc(regions[0], FID_X);
  const FieldAccessor<WRITE_DISCARD,int,1,coord_t,
          Realm::AffineAccessor<int,1,coord_t> > recvacc(regions[1], FID_Z);
  // const FieldAccessor<READ_ONLY,int,1,coord_t,
  //         Realm::AffineAccessor<int,1,coord_t> > mappingacc(regions[2], FID_RANK);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  const int *sendbuf = sendacc.ptr(rect.lo);
  int *recvbuf = recvacc.ptr(rect.lo);

  // Rect<1> rect_mapping = runtime->get_index_space_domain(ctx,
  //                 task->regions[2].region.get_index_space());
  // const int *mapping_ptr = mappingacc.ptr(rect_mapping.lo);

  printf("Running coll for point %d pid " IDFMT ", size %ld, index size %ld\n", 
        point, task->current_proc.id, rect.volume(), task->index_domain.get_volume());

  // for (size_t i = 0; i < rect_mapping.volume(); i++) {
  //   printf("%d ", mapping_ptr[i]);
  // }
  // printf("\n");

  // printf("send %p, recv %p\n", sendbuf, recvbuf);

  assert(global_comm->global_rank == point);

//   Coll_Comm global_comm;
//   int global_rank = point;
//   int global_comm_size = task->index_domain.get_volume();

//  #if defined (COLL_USE_MPI)
//   Coll_Create_comm(&global_comm, global_comm_size, global_rank, mapping_ptr);
// #else
//   Coll_Create_comm(&global_comm, global_comm_size, global_rank, NULL);
// #endif

//   assert(mapping_ptr[point] == global_comm.mpi_rank);
//   assert(global_comm_size == rect_mapping.volume());

  Coll_Alltoall((void*)sendbuf, task_arg.sendcount, collInt, 
                (void*)recvbuf, task_arg.sendcount, collInt,
                global_comm);
  printf("Point %d, Alltoall Done\n", point);
}

void finalize_comm_cpu_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  const int point = task->index_point.point_data[0];
  Coll_Comm *global_comm = task->futures[0].get_result<Coll_Comm*>();

  assert(global_comm->global_rank == point);
  assert(global_comm->status == true);
  
  Coll_Comm_free(global_comm);
  free(global_comm);
  global_comm = NULL;
  printf("Point %d, Finalize Done\n", point);
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

  int mpi_rank = 0;
#if defined (COLL_USE_MPI)
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
  int tid = point % task_arg.nb_threads;
  //int global_rank = mpi_rank * task_arg.nb_threads + tid;
  int global_rank = point;

  int start_value = global_rank * task_arg.sendcount;
  for (size_t i = 0; i < rect.volume(); i+= task_arg.sendcount) {
    for (int j = 0; j < task_arg.sendcount; j++) {
      if (recvbuf[i+j] != start_value + j) {
        printf("point %d, tid %d, i %ld j %d, val %d, expect %d\n", 
               point, tid, i, j, recvbuf[i+j], start_value + j);
        assert(0);
      }
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
    TaskVariantRegistrar registrar(INIT_MAPPING_TASK_ID, "init_mapping");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_mapping_task>(registrar, "init_mapping");
  }

  {
    TaskVariantRegistrar registrar(INIT_COMM_CPU_TASK_ID, "init_comm_cpu");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Coll_Comm*, init_comm_cpu_task>(registrar, "init_comm_cpu");
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

  {
    TaskVariantRegistrar registrar(FINALIZE_COMM_CPU_TASK_ID, "finalize_comm_cpu");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<finalize_comm_cpu_task>(registrar, "finalize_comm_cpu");
  }

  int val = Runtime::start(argc, argv);
#ifdef COLL_USE_MPI
  MPI_Finalize();
#endif
  return val;
}
