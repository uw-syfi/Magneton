// Attaching to torch, and detaching from it.
//
// Two things need C++ signatures rather than a C ABI. The RecordFunction enter
// callback must hand back a std::unique_ptr<at::ObserverContext>, which carries
// ownership a C boundary cannot express -- torch destroys it after the exit
// callback has run. And the device synchronize before a trace ends is a CUDA
// call reached through libtorch, so that it uses the run's device.

#include <cstdint>
#include <memory>

#include <ATen/record_function.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#elif defined(USE_ROCM)
#include <hip/hip_runtime.h>
#endif

#include "capture.h"
#include "rust_run.h"
#include "macros.h"
#include "state.h"

namespace {

// The pair torch calls on every op. Neither decides anything -- the enter
// hands the RecordFunction to Rust and returns the index it gets back, the
// exit hands that index in again.
auto OnFunctionEnter(const at::RecordFunction &func)
    -> std::unique_ptr<at::ObserverContext> {
  auto *state = eprof::ThreadLocalState::Get();
  if (state == nullptr) {
    return nullptr;
  }
  return std::make_unique<eprof::KinetoObserverContext>(
      eprof_begin_op(state->queue_, &func));
}

void OnFunctionExit(const at::RecordFunction &func, at::ObserverContext *ctx) {
  auto *state = eprof::ThreadLocalState::Get();
  if (state == nullptr || ctx == nullptr) {
    return;
  }
  eprof_end_op(state->queue_,
               static_cast<eprof::KinetoObserverContext *>(ctx)->op_index_,
               func.scope() == at::RecordScope::USER_SCOPE ? 1 : 0);
}

}  // namespace

extern "C" {

void eprof_callbacks_push(void) {
  auto *state = eprof::ThreadLocalState::Get();
  EPROF_CHECK(state, "Expected profiler state set");
  // No scope filter: every scope is recorded, and the profiler has never
  // offered a way to narrow it.
  // setCallbackHandle rather than assigning handle_: it is the base class's
  // now, and protected.
  state->setCallbackHandle(at::addThreadLocalCallback(
      at::RecordFunctionCallback(OnFunctionEnter, OnFunctionExit)
          .needsInputs(eprof_run_records_shapes())));
}

void eprof_callbacks_remove(void) {
  auto *state = eprof::ThreadLocalState::Get();
  if (state != nullptr) {
    state->removeCallback();
  }
}

// c10's raw clock reading. The run's converter is calibrated against this
// one, so a stamp taken anywhere else would not convert correctly.
int64_t eprof_approx_now(void) {
  return static_cast<int64_t>(c10::getApproximateTime());
}

// at::RecordFunction's per-thread counter, which is what a subqueue is keyed
// on. Not the OS thread id, and not something Rust can compute.
uint64_t eprof_current_thread_id(void) {
  return at::RecordFunction::currentThreadId();
}

void eprof_device_synchronize(void) {
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#elif defined(USE_ROCM)
  hipDeviceSynchronize();
#endif
}

}  // extern "C"
