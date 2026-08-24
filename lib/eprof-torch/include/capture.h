#pragma once


#include <cstdint>

#include <ATen/record_function.h>

namespace eprof {

// Carried between the RecordFunction enter and exit callbacks: where the op
// is in the Rust op store, so the exit can complete it.
struct KinetoObserverContext : public at::ObserverContext {
  explicit KinetoObserverContext(size_t op_index) : op_index_{op_index} {}
  size_t op_index_;
};

}  // namespace eprof

extern "C" {

// The scalar fields of a RecordFunction, filled in one call. `name` points
// into the RecordFunction and is valid for the callback's duration.
typedef struct {
  const char *name;
  int64_t sequence_number;
  uint64_t forward_tid;
  uint8_t scope;
  uint64_t record_function_id;
  int32_t is_nccl_meta;
  int32_t is_user_scope;
} EprofOpFields;

void eprof_rf_read(const void *fn, EprofOpFields *out);

// The three reads that need a c10::IValue. Each walks the RecordFunction and
// pushes straight into the Rust stream it is given.
void eprof_rf_push_inputs(const void *fn, void *inputs);
void eprof_rf_push_kwinputs(const void *fn, void *subqueue);
void eprof_rf_push_nccl_meta(const void *fn, void *subqueue);

}  // extern "C"
