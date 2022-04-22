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
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  ALLGATHER_TASK_ID,
};

enum FieldIDs {
  FID_X,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_subregions = 2;
  int count_per_subregion = 80;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-n"))
        count_per_subregion = atoi(command_args.argv[++i]);
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

  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  runtime->attach_name(input_lr, "input_lr");

  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  runtime->attach_name(ip, "ip");

  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp, "input_lp");

  // Create our launch domain.  Note that is the same as color domain
  // as we are going to launch one task for each subregion we created.
  ArgumentMap arg_map;
  int print_flag = 1;
  {
    IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
                                TaskArgument(&print_flag, sizeof(int)), arg_map);
    init_launcher.add_region_requirement(
        RegionRequirement(input_lp, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr));
    init_launcher.region_requirements[0].add_field(FID_X);
    runtime->execute_index_space(ctx, init_launcher);
  }

  {
    IndexLauncher allgather_launcher(ALLGATHER_TASK_ID, color_is, 
                                        TaskArgument(&print_flag, sizeof(int)), arg_map);
    allgather_launcher.add_region_requirement(
        RegionRequirement(input_lr, 0/*projection ID*/, 
                            READ_ONLY, EXCLUSIVE, input_lr));
    allgather_launcher.region_requirements[0].add_field(FID_X);
    runtime->execute_index_space(ctx, allgather_launcher);
  }

  print_flag = 0;
  runtime->issue_execution_fence(ctx);
  Future f_start = runtime->get_current_time_in_microseconds(ctx);
  double ts_start = f_start.get_result<long long>();

  for (int i = 0; i < 8; i++) {
    {
      IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
                                  TaskArgument(&print_flag, sizeof(int)), arg_map);
      init_launcher.add_region_requirement(
          RegionRequirement(input_lp, 0/*projection ID*/, 
                            READ_WRITE, EXCLUSIVE, input_lr));
      init_launcher.region_requirements[0].add_field(FID_X);
      runtime->execute_index_space(ctx, init_launcher);
    }

    {
      IndexLauncher allgather_launcher(ALLGATHER_TASK_ID, color_is, 
                                          TaskArgument(&print_flag, sizeof(int)), arg_map);
      allgather_launcher.add_region_requirement(
          RegionRequirement(input_lr, 0/*projection ID*/, 
                              READ_ONLY, EXCLUSIVE, input_lr));
      allgather_launcher.region_requirements[0].add_field(FID_X);
      runtime->execute_index_space(ctx, allgather_launcher);
    }
  }

  runtime->issue_execution_fence(ctx);
  Future f_end = runtime->get_current_time_in_microseconds(ctx);
  double ts_end = f_end.get_result<long long>();

  double sim_time = 1e-3 * (ts_end - ts_start);
  printf("ELAPSED TIME = %.3f ms\n", sim_time);


  runtime->destroy_logical_region(ctx, input_lr);
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

  const FieldAccessor<WRITE_DISCARD,float,1,coord_t,
        Realm::AffineAccessor<float,1,coord_t> > acc(regions[0], fid);
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  float* ptr = acc.ptr(rect.lo);
  if (print_flag) {
    printf("Initializing field %d for block %d, size %ld, pid " IDFMT "\n", fid, point, rect.volume(), task->current_proc.id);
  }

}

void allgather_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  const int print_flag = *((const int*)task->args);

  const FieldAccessor<WRITE_DISCARD,float,1,coord_t,
        Realm::AffineAccessor<float,1,coord_t> > acc(regions[0], fid);
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  float* ptr = acc.ptr(rect.lo);
  if (print_flag) {
    printf("allgather field %d for block %d, buf %p, pid " IDFMT "\n", fid, point, ptr, task->current_proc.id);
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

  int val = Runtime::start(argc, argv);
  return val;
}
