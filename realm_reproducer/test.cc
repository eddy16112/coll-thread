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
#include "legion.h"
#include "default_mapper.h"

// #include "legate_c.h"

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 100,
  INIT_FIELD_TASK_ID,
  ALLGATHER_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
};

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

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_subregions = 2;
  int count_per_subregion = 8;
  int num_iterations = 5;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-n"))
        count_per_subregion = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-i"))
        num_iterations = atoi(command_args.argv[++i]);
    }
  }

  int num_elements = count_per_subregion * num_subregions; 

  // Create our logical regions using the same schemas as earlier examples
  Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect); 
  runtime->attach_name(is, "is");
  FieldSpace input_fs = runtime->create_field_space(ctx);
  runtime->attach_name(input_fs, "input_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(float),FID_X);
    runtime->attach_name(input_fs, FID_X, "X");
  }

  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  runtime->attach_name(ip, "ip");

  LogicalRegion input_lr[2];
  LogicalPartition input_lp[2];
  for (int i = 0; i < 2; i++) {
    input_lr[i] = runtime->create_logical_region(ctx, is, input_fs);
    // runtime->attach_name(input_lr[i], "input1_lr");
    input_lp[i] = runtime->get_logical_partition(ctx, input_lr[i], ip);
    // runtime->attach_name(input_lp[i], "input1_lp");
  }

  ArgumentMap arg_map;
  int print_flag = 1;
  for (int i = 0; i < 2; i++) {
    {
      IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
                                  TaskArgument(&print_flag, sizeof(int)), arg_map);
      init_launcher.add_region_requirement(
          RegionRequirement(input_lp[i], 0/*projection ID*/, 
                            WRITE_DISCARD, EXCLUSIVE, input_lr[i]));
      init_launcher.region_requirements[0].add_field(FID_X);
      runtime->execute_index_space(ctx, init_launcher);
    }
    print_flag = 0;
  }

  runtime->issue_execution_fence(ctx);
  Future f_start = runtime->get_current_time_in_microseconds(ctx);
  double ts_start = f_start.get_result<long long>();

  for (int i = 0; i < num_iterations; i++) {
    // {
    //   IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
    //                               TaskArgument(&print_flag, sizeof(int)), arg_map);
    //   init_launcher.add_region_requirement(
    //       RegionRequirement(in_lp, 0/*projection ID*/, 
    //                         READ_WRITE, EXCLUSIVE, in_lr));
    //   init_launcher.region_requirements[0].add_field(FID_X);
    //   runtime->execute_index_space(ctx, init_launcher);
    // }

    {
      IndexLauncher allgather_launcher(ALLGATHER_TASK_ID, color_is, 
                                          TaskArgument(&print_flag, sizeof(int)), arg_map);
      allgather_launcher.add_region_requirement(
          RegionRequirement(input_lr[i % 2], 0/*projection ID*/, 
                            READ_ONLY, EXCLUSIVE, input_lr[i % 2]));
      allgather_launcher.region_requirements[0].add_field(FID_X);
      allgather_launcher.add_region_requirement(
        RegionRequirement(input_lp[(i+1) % 2], 0/*projection ID*/, 
                          READ_WRITE, EXCLUSIVE, input_lr[(i+1) % 2]));
      allgather_launcher.region_requirements[1].add_field(FID_X);
      runtime->execute_index_space(ctx, allgather_launcher);
    }
    runtime->issue_execution_fence(ctx);
  }

  runtime->issue_execution_fence(ctx);
  Future f_end = runtime->get_current_time_in_microseconds(ctx);
  double ts_end = f_end.get_result<long long>();

  double sim_time = 1e-3 * (ts_end - ts_start);
  printf("ELAPSED TIME = %.3f ms\n", sim_time);

#if 0
  {
    IndexLauncher check_launcher(CHECK_TASK_ID, color_is, 
                                TaskArgument(NULL, 0), arg_map);
    check_launcher.add_region_requirement(
        RegionRequirement(input_lp[0], 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr[0]));
    check_launcher.region_requirements[0].add_field(FID_X);
    runtime->execute_index_space(ctx, check_launcher);
  }
#endif

  for (int i = 0; i < 2; i++) {
    runtime->destroy_logical_region(ctx, input_lr[i]);
  }
  runtime->destroy_field_space(ctx, input_fs);
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
  const int print_flag = *((const int*)task->args);

  const FieldAccessor<READ_WRITE,float,1,coord_t,
        Realm::AffineAccessor<float,1,coord_t> > acc(regions[0], fid);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  float* ptr = acc.ptr(rect.lo);
  if (print_flag) {
    printf("Initializing field %d for block %d, size %ld, pid " IDFMT "\n", fid, point, rect.volume(), task->current_proc.id);
  }
#ifndef DRY_RUN
  usleep(200000);
#endif

#if 0
  for (size_t i = 0; i < rect.volume(); i++) {
    ptr[i] = static_cast<float>(point);
  }
#endif

}

void allgather_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2); 
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  const int print_flag = *((const int*)task->args);

  const FieldAccessor<READ_ONLY,float,1,coord_t,
        Realm::AffineAccessor<float,1,coord_t> > acc_in(regions[0], fid);
  const FieldAccessor<READ_WRITE,float,1,coord_t,
        Realm::AffineAccessor<float,1,coord_t> > acc_out(regions[1], fid);

  Rect<1> rect_in = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  Rect<1> rect_out = runtime->get_index_space_domain(ctx,
                  task->regions[1].region.get_index_space());
  if (print_flag) {
    printf("allgather field %d for block %d, pid " IDFMT "\n", fid, point, task->current_proc.id);
  }

#ifndef DRY_RUN
  usleep(200000);
#endif

 #if 0
  const float* ptr_in = acc_in.ptr(rect_in.lo);
  float* ptr_out = acc_out.ptr(rect_out.lo);
  int nb_ranks = task->index_domain.get_volume();

  assert(rect_in.volume() == rect_out.volume() * nb_ranks);
  for (size_t i = 0; i < rect_out.volume(); i++) {
    for (int j = 0; j < nb_ranks; j++) {
      ptr_out[i] += ptr_in[i + j * rect_out.volume()];
    }
  }
 #endif 

}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  const int point = task->index_point.point_data[0];

  const FieldAccessor<READ_ONLY,float,1,coord_t,
        Realm::AffineAccessor<float,1,coord_t> > acc(regions[0], FID_X);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());

  const float *ptr = acc.ptr(rect.lo);

  printf("point %d ", point);
  for (size_t i = 0; i < rect.volume(); i++) {
    printf("%.0f, ", ptr[i]);
  }
  printf("\n");

  printf("Point %d SUCCESS!\n", point);
}

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


int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }

  {
    TaskVariantRegistrar registrar(ALLGATHER_TASK_ID, "allgather");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<allgather_task>(registrar, "allgather");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  Runtime::add_registration_callback(mapper_registration);

  int val = Runtime::start(argc, argv);
  return val;
}
