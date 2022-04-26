#pragma once

// #include "cunumeric/cunumeric.h"

#include "core/mapping/base_mapper.h"

class CuNumericMapper : public legate::mapping::BaseMapper {
 public:
  CuNumericMapper(Legion::Runtime* rt,
                  Legion::Machine machine,
                  const legate::LibraryContext& context);
  virtual ~CuNumericMapper(void) {}

 private:
  CuNumericMapper(const CuNumericMapper& rhs) = delete;
  CuNumericMapper& operator=(const CuNumericMapper& rhs) = delete;

  // Legate mapping functions
 public:
  virtual bool is_pure() const override { return true; }
  virtual legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override;
  virtual std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override;
  virtual legate::Scalar tunable_value(legate::TunableID tunable_id) override;

 private:
  const int32_t min_gpu_chunk;
  const int32_t min_cpu_chunk;
  const int32_t min_omp_chunk;
  const int32_t eager_fraction;
};