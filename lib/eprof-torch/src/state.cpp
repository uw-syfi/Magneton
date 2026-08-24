// The object torch reports allocations into, and the run's clock.

#include "state.h"

#include <functional>
#include <memory>
#include <utility>

#include <c10/util/ThreadLocalDebugInfo.h>

#include "rust_queue.h"
#include "rust_run.h"
#include "macros.h"

namespace eprof {

ThreadLocalState *ThreadLocalState::Get() {
  return static_cast<ThreadLocalState *>(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE));
}

void ThreadLocalState::reportMemoryUsage(void *ptr, int64_t alloc_size,
                                         size_t total_allocated,
                                         size_t total_reserved,
                                         c10::Device device) {
  // Guarded here as well as by memoryProfilingEnabled(): torch does not
  // consult that on every path into the reporter.
  if (!memoryProfilingEnabled()) {
    return;
  }
  eprof_queue_report_alloc(static_cast<EprofRecordQueue *>(queue_),
                           reinterpret_cast<uint64_t>(ptr), alloc_size,
                           total_allocated, total_reserved,
                           static_cast<int8_t>(device.type()),
                           static_cast<int8_t>(device.index()));
}

void ThreadLocalState::reportOutOfMemory(int64_t alloc_size,
                                         size_t total_allocated,
                                         size_t total_reserved,
                                         c10::Device device) {
  if (!memoryProfilingEnabled()) {
    return;
  }
  eprof_queue_report_oom(static_cast<EprofRecordQueue *>(queue_), alloc_size,
                         total_allocated, total_reserved,
                         static_cast<int8_t>(device.type()),
                         static_cast<int8_t>(device.index()));
}

namespace {
std::function<c10::time_t(c10::approx_time_t)> installed_converter;
}  // namespace

}  // namespace eprof

extern "C" {

void *eprof_state_push(void *queue, int32_t profile_memory) {
  EPROF_CHECK(eprof::ThreadLocalState::Get() == nullptr,
              "Profiler is already enabled on this thread.");
  auto state = std::make_shared<eprof::ThreadLocalState>(
      queue, profile_memory != 0);
  auto *raw = state.get();
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE,
                                   std::move(state));
  return raw;
}

void eprof_state_pop(void) {
  auto *state = eprof::ThreadLocalState::Get();
  EPROF_CHECK(state, "Can't disable the profiler when it's not running");
  // Built before the state goes: an op whose exit callback never ran keeps the
  // sentinel, so materialization can borrow an end time from its parent.
  eprof::installed_converter =
      [raw = state->clock_.makeConverter()](c10::approx_time_t t) {
        return t == std::numeric_limits<c10::approx_time_t>::min()
                   ? std::numeric_limits<c10::time_t>::min()
                   : raw(t);
      };
  state->removeCallback();
  c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);
}

int64_t eprof_convert_time(void * /*ctx*/, int64_t t) {
  const auto &fn = eprof::installed_converter;
  return fn ? static_cast<int64_t>(fn(t)) : t;
}

int32_t eprof_state_active(void) {
  return eprof::ThreadLocalState::Get() != nullptr ? 1 : 0;
}

}  // extern "C"
