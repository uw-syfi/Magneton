
#include <ATen/core/TensorBase.h>
#include <Python.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Logging.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <frameobject.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/python.h>
#include <torch/utils.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "state.h"
#include "tracer.h"
#include "rust_py_cache.h"
#include "rust_queue.h"
#include "rust_run.h"

namespace py = pybind11;

namespace {

// Records a python frame entry on the calling thread's subqueue.
inline void PushPyCall(void *queue, uint64_t key, c10::approx_time_t t) {
  eprof_queue_push_py_call(queue, key, static_cast<int64_t>(t));
}

// A callsite's identity in the Rust cache (eprof-python/src/cache.rs).
// Zero means "not interned".
using TraceKey = uint64_t;

// The clock is the caller's; a reading crosses back through this.
using ConvertFn = int64_t (*)(void *ctx, int64_t t);


EprofCodeLoc frameLoc(PyFrameObject *frame) {
  auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
  const auto filename = THPUtils_unpackStringView(code->co_filename);
  const auto name = THPUtils_unpackStringView(code->co_name);
  return EprofCodeLoc{reinterpret_cast<uint64_t>(filename.data()),
                      reinterpret_cast<uint64_t>(name.data()),
                      PyFrame_GetLineNumber(frame)};
}

void trimPrefixes(EprofPyCache *cache) {
  static const auto prefixes = []() {
    pybind11::gil_scoped_acquire gil;
    return py::module::import("torch.profiler.python_tracer")
        .attr("_prefix_regex")()
        .cast<std::vector<std::string>>();
  }();
  std::vector<const char *> ptrs;
  ptrs.reserve(prefixes.size());
  for (const auto &p : prefixes) {
    ptrs.push_back(p.c_str());
  }
  eprof_pycache_trim_prefixes(cache, ptrs.data(), ptrs.size());
}

PyCodeObject *moduleCallCode() {
  static auto *module_call_code = []() {
    pybind11::gil_scoped_acquire gil;
    auto *res = py::module::import("torch.nn")
                    .attr("Module")
                    .attr("__call__")
                    .attr("__code__")
                    .ptr();
    TORCH_INTERNAL_ASSERT(PyCode_Check(res));
    return (PyCodeObject *)res;
  }();
  return module_call_code;
}

struct ThreadLocalResults;
struct TraceContext {
  PyObject_HEAD ThreadLocalResults *thread_local_results_;
};

// CPython boilerplate to define `TraceContext` as a proper python object.
static PyTypeObject TraceContextType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "TraceContext", /* tp_name */
    sizeof(TraceContext),                             /* tp_basicsize */
    0,                                                /* tp_itemsize */
    nullptr,                                          /* tp_dealloc */
    0,
    /* tp_vectorcall_offset */
    nullptr,             /* tp_getattr */
    nullptr,             /* tp_setattr */
    nullptr,             /* tp_reserved */
    nullptr,             /* tp_repr */
    nullptr,             /* tp_as_number */
    nullptr,             /* tp_as_sequence */
    nullptr,             /* tp_as_mapping */
    nullptr,             /* tp_hash  */
    nullptr,             /* tp_call */
    nullptr,             /* tp_str */
    nullptr,             /* tp_getattro */
    nullptr,             /* tp_setattro */
    nullptr,             /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,  /* tp_flags */
    "Python tracer TLS", /* tp_doc */
    nullptr,             /* tp_traverse */
    nullptr,             /* tp_clear */
    nullptr,             /* tp_richcompare */
    0,                   /* tp_weaklistoffset */
    nullptr,             /* tp_iter */
    nullptr,             /* tp_iternext */
    nullptr,             /* tp_methods */
    nullptr,             /* tp_members */
    nullptr,             /* tp_getset */
    nullptr,             /* tp_base */
    nullptr,             /* tp_dict */
    nullptr,             /* tp_descr_get */
    nullptr,             /* tp_descr_set */
    0,                   /* tp_dictoffset */
    nullptr,             /* tp_init */
    nullptr,             /* tp_alloc */
    PyType_GenericNew,   /* tp_new */
    nullptr              /* tp_free */
};

class gil_and_restore_thread {
 public:
  gil_and_restore_thread()
      : gil_(), initial_thread_state_{PyThreadState_Get()} {}
  ~gil_and_restore_thread() {
    PyThreadState_Swap(initial_thread_state_);

    // `gil_scoped_acquire` is a bit fragile in on-demand mode:
    // https://github.com/pytorch/pytorch/pull/91684#issuecomment-1413154458
    if (!Py_IsInitialized()) {
      gil_.disarm();
    }
  }

  PyThreadState *initial_thread_state() const { return initial_thread_state_; }

 private:
  pybind11::gil_scoped_acquire gil_;
  PyThreadState *initial_thread_state_;
};

// --- Thread local cache ---
class PythonTracer;
struct ThreadLocalResults {
  ThreadLocalResults(PyThreadState *thread_state, size_t python_tid,
                     PythonTracer *active_tracer)
      : thread_state_{thread_state},
        ctx_{(TraceContext *)TraceContextType.tp_alloc(&TraceContextType, 0)},
        python_tid_{python_tid},
        active_tracer_{active_tracer} {
    ctx_->thread_local_results_ = this;
  }

  ThreadLocalResults() = delete;
  ThreadLocalResults(const ThreadLocalResults &) = delete;
  ThreadLocalResults(ThreadLocalResults &&) = delete;
  ThreadLocalResults &operator=(const ThreadLocalResults &) = delete;
  ThreadLocalResults &operator=(const ThreadLocalResults &&) = delete;

  ~ThreadLocalResults() { Py_DECREF((PyObject *)ctx_); }

  PyThreadState *thread_state_;
  TraceContext *ctx_;
  // Its own index among the tracer's threads. Part of every cache key, so two
  // threads at one callsite get two keys and replay on separate stacks.
  size_t python_tid_;
  PythonTracer *active_tracer_;
  // Just the times; which frame each one closes is worked out in replay.
  std::vector<c10::approx_time_t> exit_times_;
  std::vector<c10::approx_time_t> c_exit_times_;
};

// --- Tracing implementation ---
class PythonTracer final {
 public:
  explicit PythonTracer(void *queue);
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~PythonTracer();

  static int pyProfileFn(PyObject *obj, PyFrameObject *frame, int what,
                         PyObject *arg);

  void stop();
  void getEvents(void *events, void *queue, ConvertFn convert, void *ctx,
                 int64_t end_time_ns);

  struct StartFrame {
    TraceKey trace_key_;
    c10::approx_time_t start_time{};
  };

 private:
  void recordPyCall(ThreadLocalResults &tls, PyFrameObject *frame,
                    bool is_startup_frame);

  void recordCCall(ThreadLocalResults &tls, PyFrameObject *frame,
                   PyObject *arg);

  const std::vector<PyThreadState *> interpreterThreads() const;

  std::atomic<bool> active_lock_{false};
  bool active_{false};

  void *queue_;  // EprofRecordQueue*
  PyInterpreterState *interpreter_{nullptr};
  PyCodeObject *module_call_code_;
  EprofPyCache *cache_;

  std::vector<StartFrame> start_frames_;
  std::deque<ThreadLocalResults> thread_local_results_;
};

const std::vector<PyThreadState *> PythonTracer::interpreterThreads() const {
  pybind11::gil_scoped_acquire gil;
  std::vector<PyThreadState *> out;
  if (interpreter_) {
    auto *thread_state = PyInterpreterState_ThreadHead(interpreter_);
    while (thread_state != nullptr) {
      out.push_back(thread_state);
      thread_state = PyThreadState_Next(thread_state);
    }
  }
  return out;
}

PythonTracer::PythonTracer(void *queue)
    : queue_(queue),

      module_call_code_(moduleCallCode()),
      cache_(eprof_pycache_create()) {
  TORCH_INTERNAL_ASSERT(queue_ != nullptr, "Queue is null");

  bool expected{false};
  active_ = active_lock_.compare_exchange_strong(expected, true);
  if (!active_) {
    TORCH_WARN(
        "There is already an active Python tracer. "
        "Refusing to register profile functions.");
    return;
  }

  gil_and_restore_thread gil;
  interpreter_ = PyInterpreterState_Get();

  if (!gil.initial_thread_state()) {
    TORCH_WARN("PyThreadState_Get returned NULL");
    return;
  }

  // Register the tracer in each thread.
  for (const auto thread_state : interpreterThreads()) {
    PyThreadState_Swap(thread_state);

    thread_local_results_.emplace_back(thread_state,
                                       thread_local_results_.size(), this);
    auto *ctx = thread_local_results_.back().ctx_;


    std::vector<THPFrameObjectPtr> current_stack;
    auto frame = PyEval_GetFrame();
    Py_XINCREF(frame);

    size_t depth = 0;  // Make sure we can't infinite loop.
    while (frame != nullptr) {
      current_stack.emplace_back(frame);
      if (++depth == 128) {
        break;
      }

      // NB: `PyFrame_GetBack` returns a strong reference.
      frame = PyFrame_GetBack(frame);
    }

    for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
      recordPyCall(thread_local_results_.back(), it->get(), true);
      auto frame_refcount = Py_REFCNT(it->get());

      // We hold one reference in `current_stack`, and the interpreter holds
      // another.
      TORCH_INTERNAL_ASSERT(frame_refcount >= 2, "Frame refcount is less than 2");
    }

    PyEval_SetProfile(PythonTracer::pyProfileFn, (PyObject *)ctx);
  }
}

void PythonTracer::stop() {
  gil_and_restore_thread gil;
  if (active_) {
    for (const auto thread_state : interpreterThreads()) {
      if (thread_state->c_profilefunc == &PythonTracer::pyProfileFn) {
        PyThreadState_Swap(thread_state);
        PyEval_SetProfile(nullptr, nullptr);
      }
    }

    auto lock_returned = active_lock_.compare_exchange_strong(active_, false);
    active_ = false;
    TORCH_INTERNAL_ASSERT(lock_returned, "Failed to return python tracer lock.");
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
PythonTracer::~PythonTracer() {
  if (active_) {
    TORCH_WARN("`PythonTracer::stop()` was not called.");
    stop();
  }
  eprof_pycache_destroy(cache_);
}

void PythonTracer::recordPyCall(ThreadLocalResults &tls, PyFrameObject *frame,
                                bool is_startup_frame) {
  const auto key = [&]() -> TraceKey {
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    const bool is_module_call = code.get() == module_call_code_;

    auto back = THPFrameObjectPtr(PyFrame_GetBack(frame));
    if (is_module_call) {
      TORCH_INTERNAL_ASSERT(back != nullptr, "Back is null");
    }
    auto *caller_frame = back.get() != nullptr ? back.get() : frame;
    const auto caller = frameLoc(caller_frame);

    THPObjectPtr self;
    if (is_module_call) {
      auto locals = THPObjectPtr(PyFrame_GetLocals(frame));
      self = THPObjectPtr(PyDict_GetItemString(locals, "self"));
      Py_INCREF(self.get());
    }

    EprofSiteKey k{};
    k.call_type = is_module_call ? EPROF_PY_MODULE_CALL : EPROF_PY_CALL;
    k.python_tid = tls.python_tid_;
    k.caller = caller;
    if (is_module_call) {
      k.value_ptr = reinterpret_cast<uint64_t>(self.get());
    } else {
      k.value_loc = frameLoc(frame);
    }

    if (const auto hit = eprof_pycache_get(cache_, &k)) {
      return hit;
    }

    // A miss: describe the callsite once and let the cache keep what it needs.
    EprofSiteRecord rec{};
    rec.key = k;
    auto caller_code = THPCodeObjectPtr(PyFrame_GetCode(caller_frame));
    rec.caller_filename =
        THPUtils_unpackStringView(caller_code->co_filename).data();
    rec.caller_funcname = THPUtils_unpackStringView(caller_code->co_name).data();

    auto own_code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    rec.value_filename = THPUtils_unpackStringView(own_code->co_filename).data();
    rec.value_funcname = THPUtils_unpackStringView(own_code->co_name).data();

    std::string cls_name;
    if (is_module_call) {
      // Every module call is described by `nn.Module.__call__`'s location,
      // which is this frame's, plus the class of the instance.
      rec.module_loc = frameLoc(frame);
      auto cls = py::handle(self.get()).attr("__class__");
      rec.module_cls = reinterpret_cast<uint64_t>(cls.ptr());
      cls_name = py::str(cls.attr("__name__")).cast<std::string>();
      rec.module_cls_name = cls_name.c_str();
    }
    return eprof_pycache_intern_site(cache_, &rec);
  }();
  const auto time = c10::getApproximateTime();
  is_startup_frame
      ? start_frames_.push_back({key, time})
      : PushPyCall(queue_, key, time);
}

void PythonTracer::recordCCall(ThreadLocalResults &tls, PyFrameObject *frame,
                               PyObject *arg) {
  TORCH_INTERNAL_ASSERT(PyCFunction_Check(arg), "Arg is not a C function");
  auto *fn = reinterpret_cast<PyCFunctionObject *>(arg);

  // NB: For C calls a new frame is not created, so we use `frame` rather than
  //     `frame->f_back`.
  EprofSiteKey k{};
  k.call_type = EPROF_PY_C_CALL;
  k.python_tid = tls.python_tid_;
  k.value_ptr = reinterpret_cast<uint64_t>(fn->m_ml);
  k.caller = frameLoc(frame);

  auto key = eprof_pycache_get(cache_, &k);
  if (key == 0) {
    EprofSiteRecord rec{};
    rec.key = k;
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    rec.caller_filename = THPUtils_unpackStringView(code->co_filename).data();
    rec.caller_funcname = THPUtils_unpackStringView(code->co_name).data();
    const auto repr = py::repr(arg).cast<std::string>();
    rec.c_function_name = repr.c_str();
    key = eprof_pycache_intern_site(cache_, &rec);
  }
  PushPyCall(queue_, key, c10::getApproximateTime());
}


void PythonTracer::getEvents(void *events, void *queue, ConvertFn convert,
                             void *ctx, int64_t end_time_ns) {
  const auto time_converter = [&](c10::approx_time_t t) {
    return static_cast<c10::time_t>(convert(ctx, static_cast<int64_t>(t)));
  };
  trimPrefixes(cache_);

  // Frames that were already on the stack when profiling started have no
  // thread of their own on record; the replay recognises them by NO_TID.
  auto *q = static_cast<EprofRecordQueue *>(queue);
  for (const auto &frame : start_frames_) {
    eprof_queue_push_python_enter(q, frame.trace_key_, EPROF_NO_TID,
                                  time_converter(frame.start_time));
  }

  std::vector<EprofExitRecord> exits;
  for (size_t python_tid = 0; python_tid < thread_local_results_.size();
       ++python_tid) {
    const auto &tls = thread_local_results_[python_tid];
    for (const auto t : tls.exit_times_) {
      exits.push_back(EprofExitRecord{time_converter(t), python_tid, 0});
    }
    for (const auto t : tls.c_exit_times_) {
      exits.push_back(EprofExitRecord{time_converter(t), python_tid, 1});
    }
  }

  // Fetched after the pushes above, which may have moved the array.
  size_t n_enters = 0;
  auto *enters =
      static_cast<EprofEnterRecord *>(eprof_queue_python_enters(q, &n_enters));
  eprof_pycache_replay(cache_, events, enters, n_enters, exits.data(),
                       exits.size(), end_time_ns);
}

// --- API ---
int PythonTracer::pyProfileFn(PyObject *obj, PyFrameObject *frame, int what,
                              PyObject *arg) {
  auto &local_results =
      *reinterpret_cast<TraceContext *>(obj)->thread_local_results_;
  switch (what) {
    case PyTrace_CALL:
      local_results.active_tracer_->recordPyCall(local_results, frame, false);
      break;

    case PyTrace_C_CALL:
      local_results.active_tracer_->recordCCall(local_results, frame, arg);
      break;

    case PyTrace_EXCEPTION:
    case PyTrace_RETURN:
      local_results.exit_times_.emplace_back(c10::getApproximateTime());
      break;

    case PyTrace_C_EXCEPTION:
    case PyTrace_C_RETURN:
      local_results.c_exit_times_.emplace_back(c10::getApproximateTime());
      break;
  }
  return 0;
}

extern "C" {

void eprof_tracer_init(void) {
  pybind11::gil_scoped_acquire gil;
  TORCH_INTERNAL_ASSERT(PyType_Ready(&TraceContextType) == 0,
                        "Failed to initialize TraceContext type");
}

void *eprof_tracer_create(void *queue) { return new PythonTracer(queue); }

void eprof_tracer_stop(void *tracer) {
  if (tracer != nullptr) {
    static_cast<PythonTracer *>(tracer)->stop();
  }
}

void eprof_tracer_get_events(void *tracer, void *events, void *queue,
                             ConvertFn convert, void *ctx,
                             int64_t end_time_ns) {
  if (tracer != nullptr) {
    static_cast<PythonTracer *>(tracer)->getEvents(events, queue, convert, ctx,
                                                   end_time_ns);
  }
}

void eprof_tracer_destroy(void *tracer) {
  delete static_cast<PythonTracer *>(tracer);
}

}  // extern "C"

}  // namespace
