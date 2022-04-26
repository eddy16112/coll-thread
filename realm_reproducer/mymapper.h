#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include "legion.h"
#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

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
virtual void map_task(const MapperContext ctx,
                          const Task& task,
                          const MapTaskInput& input,
                          MapTaskOutput& output);
public:
  AddressSpace local_node;

};
