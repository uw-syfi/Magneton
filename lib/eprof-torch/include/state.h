#pragma once

// The object torch reports allocations into.
//
// This is the one place the profiler cannot be anything but a C++ class. It is
// pushed onto c10::ThreadLocalDebugInfo under DebugInfoKind::PROFILER_STATE,
// and the allocator calls the virtuals below through a base pointer. A C ABI
// can put a function behind a pointer, but not a vtable at an address torch
// already holds -- and faking one would mean committing to the Itanium layout,
// the ordering of libtorch's virtuals, and libstdc++'s shared_ptr control
// block, none of which fail loudly when they change.
//
// It derives from torch::profiler::impl::ProfilerStateBase, and that is not
// decoration. That slot is torch's own, and torch reads it back with
// ProfilerStateBase::get(), which static_casts whatever it finds. Deriving
// only from c10::MemoryReportingInfoBase -- which is enough for the allocator,
// and was what this did -- put an object of the wrong type there. Nothing
// noticed until something called that getter: applying any
// torch.autograd.Function did, read this through ProfilerStateBase's vtable,
// and died in C++ with std::bad_alloc.
//
// The config it reports is Disabled, which is true: torch's profiler is not
// running, this one is. torch::profiler::impl::profilerEnabled() therefore
// answers false and torch leaves us alone, while profile_memory stays set so
// the allocator still calls reportMemoryUsage below.
//
// It stays thin otherwise: a queue handle and the run's clock. It answers
// torch's calls by forwarding them; it decides nothing.

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
