#pragma once


#include <cstdint>

#include <ATen/record_function.h>
#include <c10/core/Device.h>
#include <c10/util/ApproximateClock.h>
#include <torch/csrc/profiler/orchestration/observer.h>

namespace eprof {

struct ThreadLocalState : public torch::profiler::impl::ProfilerStateBase {
  ThreadLocalState(void *queue, bool profile_memory)
      : ProfilerStateBase(torch::profiler::impl::ProfilerConfig(
            // Disabled, so torch does not mistake this for its own profiler
            // running; profile_memory, so the allocator still reports here.
            torch::profiler::impl::ProfilerState::Disabled,
            /*report_input_shapes=*/false, profile_memory)),
        queue_(queue) {}

  // The state on this thread, or null when no run is in progress.
  static ThreadLocalState *Get();

  // Which profiler torch would be looking at. None of them: this is not one of
  // torch's, and saying so is what keeps torch from trying to drive it.
  torch::profiler::impl::ActiveProfilerType profilerType() override {
    return torch::profiler::impl::ActiveProfilerType::NONE;
  }
  void reportMemoryUsage(void *ptr, int64_t alloc_size, size_t total_allocated,
                         size_t total_reserved, c10::Device device) override;
  void reportOutOfMemory(int64_t alloc_size, size_t total_allocated,
                         size_t total_reserved, c10::Device device) override;

  void *queue_;  // EprofRecordQueue*
  // Calibrated when the run starts, read when it ends.
  c10::ApproximateClockToUnixTimeConverter clock_;
};

}  // namespace eprof
