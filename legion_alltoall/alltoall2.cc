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
#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

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

#define COLL_DTYPE collInt
typedef int DTYPE;

class MyMapper : public DefaultMapper {
public:
  MyMapper(Machine machine,
      Runtime *rt, Processor local);
  virtual void select_task_sources(const MapperContext ctx,
                                   const Task& task,
                                   const SelectTaskSrcInput& input,
                                         SelectTaskSrcOutput& output);
  void legate_select_sources(const MapperContext ctx,
                                    const PhysicalInstance& target,
                                    const std::vector<PhysicalInstance>& sources,
                                    std::deque<PhysicalInstance>& ranking);
public:
  AddressSpace local_node;

};

MyMapper::MyMapper(Machine m,
                  Runtime *rt,
                  Processor p)
  : DefaultMapper(rt->get_mapper_runtime(), m, p)
{
  Processor proc = Processor::get_executing_processor();
  local_node = proc.address_space();
}

void MyMapper::select_task_sources(const MapperContext ctx,
                                   const Task& task,
                                   const SelectTaskSrcInput& input,
                                         SelectTaskSrcOutput& output)
{
  legate_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

void MyMapper::legate_select_sources(const MapperContext ctx,
                                    const PhysicalInstance& target,
                                    const std::vector<PhysicalInstance>& sources,
                                    std::deque<PhysicalInstance>& ranking)
{
  std::map<Memory, uint32_t /*bandwidth*/> source_memories;
  // For right now we'll rank instances by the bandwidth of the memory
  // they are in to the destination, we'll only rank sources from the
  // local node if there are any
  bool all_local = false;
  // TODO: consider layouts when ranking source to help out the DMA system
  Memory destination_memory = target.get_location();
  std::vector<MemoryMemoryAffinity> affinity(1);
  // fill in a vector of the sources with their bandwidths and sort them
  std::vector<std::pair<PhysicalInstance, uint32_t /*bandwidth*/>> band_ranking;
  for (uint32_t idx = 0; idx < sources.size(); idx++) {
    const PhysicalInstance& instance = sources[idx];
    Memory location                  = instance.get_location();
    if (location.address_space() == local_node) {
      if (!all_local) {
        source_memories.clear();
        band_ranking.clear();
        all_local = true;
      }
    } else if (all_local)  // Skip any remote instances once we're local
      continue;
    auto finder = source_memories.find(location);
    if (finder == source_memories.end()) {
      affinity.clear();
      machine.get_mem_mem_affinity(
        affinity, location, destination_memory, false /*not just local affinities*/);
      uint32_t memory_bandwidth = 0;
      if (!affinity.empty()) {
        assert(affinity.size() == 1);
        memory_bandwidth = affinity[0].bandwidth;
#if 0
          } else {
            // TODO: More graceful way of dealing with multi-hop copies
            logger.warning("Legate mapper is potentially "
                              "requesting a multi-hop copy between memories "
                              IDFMT " and " IDFMT "!", location.id,
                              destination_memory.id);
#endif
      }
      source_memories[location] = memory_bandwidth;
      band_ranking.push_back(std::pair<PhysicalInstance, uint32_t>(instance, memory_bandwidth));
    } else
      band_ranking.push_back(std::pair<PhysicalInstance, uint32_t>(instance, finder->second));
  }
  assert(!band_ranking.empty());
  // Easy case of only one instance
  if (band_ranking.size() == 1) {
    ranking.push_back(band_ranking.begin()->first);
    return;
  }
  // Sort them by bandwidth
  std::sort(band_ranking.begin(), band_ranking.end(), physical_sort_func);
  // Iterate from largest bandwidth to smallest
  for (auto it = band_ranking.rbegin(); it != band_ranking.rend(); ++it)
    ranking.push_back(it->first);
}


void mapper_registration(Machine machine, Runtime *rt,
                          const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new MyMapper(machine, rt, *it), *it);
  }
}

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
#ifdef LEGATE_USE_GASNET
  int mpi_rank, mpi_comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
  int num_subregions = nb_threads * (mpi_comm_size-missing_nodes);
  //int num_subregions = nb_threads * mpi_comm_size -missing_nodes;
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
    allocator.allocate_field(sizeof(DTYPE),FID_X);
    runtime->attach_name(input_fs, FID_X, "X");
  }
  FieldSpace output_fs = runtime->create_field_space(ctx);
  runtime->attach_name(output_fs, "output_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(DTYPE),FID_Z);
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

  IndexLauncher init_mapping_launcher(INIT_MAPPING_TASK_ID, color_is, 
                                      TaskArgument(NULL, 0), arg_map);
  FutureMap mapping_table_future_map = runtime->execute_index_space(ctx, init_mapping_launcher);

  // mapping_table_future_map.wait_all_results();
  // size_t mapping_table_size = sizeof(int) * num_subregions;
  // int *mapping_table = (int *)malloc(mapping_table_size);
  // for (int i = 0; i < num_subregions; i++) {
  //   mapping_table[i] = mapping_table_future_map.get_result<int>(i);
  // }


#if 0
  IndexLauncher init_comm_cpu_launcher(INIT_COMM_CPU_TASK_ID, color_is, 
                                      TaskArgument(mapping_table, mapping_table_size), arg_map);
#else
  IndexLauncher init_comm_cpu_launcher(INIT_COMM_CPU_TASK_ID, color_is, 
                                      TaskArgument(NULL, 0), arg_map);
  // Future mapping_table_f = Future::from_untyped_pointer(runtime, mapping_table, sizeof(int)*num_subregions);
  for (int i = 0; i < num_subregions; i++) {
    init_comm_cpu_launcher.add_future(mapping_table_future_map.get_future(i));
  }                                   
#endif
  FutureMap comm_future_map = runtime->execute_index_space(ctx, init_comm_cpu_launcher);
  // free(mapping_table);

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

  const FieldAccessor<WRITE_DISCARD,DTYPE,1,coord_t,
        Realm::AffineAccessor<DTYPE,1,coord_t> > acc(regions[0], fid);
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  DTYPE* ptr = acc.ptr(rect.lo);
  //printf("Initializing field %d for block %d, size %ld, pid " IDFMT "\n", fid, point, rect.volume(), task->current_proc.id);

#if 0
  for (size_t i = 0; i < rect.volume(); i++) {
    ptr[i] = (DTYPE)i;
  }
#else
  for (size_t i = 0; i < rect.volume(); i++) {
    ptr[i] = (DTYPE)(point * rect.volume() + i);
  }
#endif
}

int init_mapping_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  int mpi_rank = 0;
#if defined (LEGATE_USE_GASNET)
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
  return mpi_rank;
}

Coll_Comm* init_comm_cpu_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  const int point = task->index_point[0];
#if 0
  const int* mapping_table = (const int*)task->args;
#else
#endif
  printf("Init Comm for point %d pid " IDFMT ", index size %ld\n", 
        point, task->current_proc.id, task->index_domain.get_volume());

  // for (size_t i = 0; i < rect_mapping.volume(); i++) {
  //   printf("%d ", mapping_ptr[i]);
  // }
  // printf("\n");

  Coll_Comm *global_comm = (Coll_Comm*)malloc(sizeof(Coll_Comm));
  int global_rank = point;
  int global_comm_size = task->index_domain.get_volume();
  assert(task->futures.size() == static_cast<size_t>(global_comm_size));
  int *mapping_table = (int *)malloc(sizeof(int) * global_comm_size);
  for (int i = 0; i < global_comm_size; i++) {
    const int* mapping_table_element = (const int*)task->futures[i].get_buffer(Memory::SYSTEM_MEM);
    mapping_table[i] = *mapping_table_element;
    //printf("%d ", mapping_table[i]);
  }
  //printf("\n");

 #if defined (LEGATE_USE_GASNET)
  collCommCreate(global_comm, global_comm_size, global_rank, mapping_table);
#else
  collCommCreate(global_comm, global_comm_size, global_rank, NULL);
#endif

  assert(mapping_table[point] == global_comm->mpi_rank);
  // assert(global_comm_size == rect_mapping.volume());
  free(mapping_table);
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

  const FieldAccessor<READ_ONLY,DTYPE,1,coord_t,
          Realm::AffineAccessor<DTYPE,1,coord_t> > sendacc(regions[0], FID_X);
  const FieldAccessor<WRITE_DISCARD,DTYPE,1,coord_t,
          Realm::AffineAccessor<DTYPE,1,coord_t> > recvacc(regions[1], FID_Z);
  // const FieldAccessor<READ_ONLY,int,1,coord_t,
  //         Realm::AffineAccessor<int,1,coord_t> > mappingacc(regions[2], FID_RANK);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  const DTYPE *sendbuf = sendacc.ptr(rect.lo);
  DTYPE *recvbuf = recvacc.ptr(rect.lo);

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

//  #if defined (LEGATE_USE_GASNET)
//   Coll_Create_comm(&global_comm, global_comm_size, global_rank, mapping_ptr);
// #else
//   Coll_Create_comm(&global_comm, global_comm_size, global_rank, NULL);
// #endif

//   assert(mapping_ptr[point] == global_comm.mpi_rank);
//   assert(global_comm_size == rect_mapping.volume());

  collAlltoall((void*)sendbuf, task_arg.sendcount, COLL_DTYPE, 
                (void*)recvbuf, task_arg.sendcount, COLL_DTYPE,
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
  
  collCommDestroy(global_comm);
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

  const FieldAccessor<READ_ONLY,DTYPE,1,coord_t,
        Realm::AffineAccessor<DTYPE,1,coord_t> > recvacc(regions[0], FID_Z);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());

  const DTYPE *recvbuf = recvacc.ptr(rect.lo);

  int mpi_rank = 0;
#if defined (LEGATE_USE_GASNET)
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
  int tid = point % task_arg.nb_threads;
  //int global_rank = mpi_rank * task_arg.nb_threads + tid;
  int global_rank = point;

#if 0
  int start_value = global_rank * task_arg.sendcount;
  for (size_t i = 0; i < rect.volume(); i+= task_arg.sendcount) {
    for (int j = 0; j < task_arg.sendcount; j++) {
      if (recvbuf[i+j] != (DTYPE)(start_value + j)) {
        printf("point %d, tid %d, i %ld j %d, val %d, expect %d\n", 
               point, tid, i, j, (int)recvbuf[i+j], start_value + j);
        assert(0);
      }
    }
  }
#else
  int ct_x = 0;
  for (int i = 0; i < rect.volume(); i += task_arg.sendcount) {
    int start_value = global_rank * task_arg.sendcount + ct_x * rect.volume();
    for (int j = 0; j < task_arg.sendcount; j++) {
      if (recvbuf[i+j] != (DTYPE)start_value + j) {
        printf("point %d, tid %d, i %d j %d, val %d, expect %d\n", 
               point, tid, i, j, (int)recvbuf[i+j], start_value + j);
        assert(0);
      }
    }
    ct_x ++;
  }
#endif
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

#ifdef LEGATE_USE_GASNET
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
    Runtime::preregister_task_variant<int, init_mapping_task>(registrar, "init_mapping");
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

  Runtime::add_registration_callback(mapper_registration);

  int val = Runtime::start(argc, argv);
#ifdef LEGATE_USE_GASNET
  MPI_Finalize();
#endif
  return val;
}
