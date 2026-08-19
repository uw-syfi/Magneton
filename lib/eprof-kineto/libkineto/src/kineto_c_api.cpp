// Implementation of the C ABI declared in include/kineto_c_api.h.
//
// Every function here forwards to libkineto::api() and does nothing else. The
// only real work is owning the two handle types: a stopped trace, and a buffer
// of activities on its way in.

#include "kineto_c_api.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

// Not a public header: the time converter hook lives next to the clock it
// belongs to, and this file is inside libkineto so it can reach it.
#include "ApproximateClock.h"

#include "ActivityTraceInterface.h"
#include "ActivityType.h"
#include "GenericTraceActivity.h"
#include "ITraceActivity.h"
#include "ThreadUtil.h"
#include "libkineto.h"
#include "time_since_epoch.h"

namespace {

// A stopped trace, plus its activity list. libkineto hands back the list as a
// pointer to a vector it owns; caching it keeps indexing cheap and lets the
// count be answered without a null check per call.
struct Trace {
  std::unique_ptr<libkineto::ActivityTraceInterface> trace;
  std::vector<const libkineto::ITraceActivity*> activities;
};

const libkineto::ITraceActivity* asActivity(const KinetoCActivity* a) {
  return reinterpret_cast<const libkineto::ITraceActivity*>(a);
}

// Only valid for activities this API produced -- see the header.
libkineto::GenericTraceActivity* asGeneric(KinetoCActivity* a) {
  return reinterpret_cast<libkineto::GenericTraceActivity*>(a);
}

} // namespace

extern "C" {

// --- Lifecycle --------------------------------------------------------------

void kineto_c_prepare_trace(
    int cpu_only,
    const int32_t* activity_types,
    size_t n_activity_types,
    const char* config) {
  libkineto::api().resetKinetoTLS();
  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(cpu_only != 0, /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }
  if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
  }

  std::set<libkineto::ActivityType> types;
  for (size_t i = 0; i < n_activity_types; ++i) {
    types.insert(static_cast<libkineto::ActivityType>(activity_types[i]));
  }
  libkineto::api().activityProfiler().prepareTrace(
      types, config != nullptr ? std::string(config) : std::string());
}

void kineto_c_start_trace(void) {
  libkineto::api().activityProfiler().startTrace();
}

KinetoCTrace* kineto_c_stop_trace(void) {
  auto trace = libkineto::api().activityProfiler().stopTrace(false, nullptr);
  if (trace == nullptr) {
    return nullptr;
  }
  auto* out = new Trace{std::move(trace), {}};
  if (const auto* list = out->trace->activities()) {
    out->activities = *list;
  }
  return reinterpret_cast<KinetoCTrace*>(out);
}

void kineto_c_toggle_collection_dynamic(int enable) {
  libkineto::api().activityProfiler().toggleCollectionDynamic(enable != 0);
}

void kineto_c_push_correlation_id(uint64_t id) {
  libkineto::api().activityProfiler().pushCorrelationId(id);
}

void kineto_c_pop_correlation_id(void) {
  libkineto::api().activityProfiler().popCorrelationId();
}

void kineto_c_push_user_correlation_id(uint64_t id) {
  libkineto::api().activityProfiler().pushUserCorrelationId(id);
}

void kineto_c_pop_user_correlation_id(void) {
  libkineto::api().activityProfiler().popUserCorrelationId();
}

void kineto_c_record_thread_info(void) {
  libkineto::api().activityProfiler().recordThreadInfo();
}

int64_t kineto_c_process_id(void) {
  return libkineto::processId();
}

int64_t kineto_c_system_thread_id(void) {
  return libkineto::systemThreadId();
}

int64_t kineto_c_time_since_epoch_now(void) {
  return libkineto::timeSinceEpoch(std::chrono::system_clock::now());
}

int kineto_c_has_collectives_profiler(void) {
#ifdef KINETO_HAS_NCCL_PROFILER
  return 1;
#else
  return 0;
#endif
}

void kineto_c_set_time_converter(
    int64_t (*convert)(void* ctx, int64_t t),
    void* ctx) {
  if (convert == nullptr) {
    libkineto::get_time_converter() = [](libkineto::approx_time_t t) {
      return static_cast<libkineto::time_t>(t);
    };
    return;
  }
  libkineto::get_time_converter() = [convert, ctx](libkineto::approx_time_t t) {
    return static_cast<libkineto::time_t>(convert(ctx, t));
  };
}

// --- Reading a finished trace -----------------------------------------------

void kineto_c_trace_free(KinetoCTrace* trace) {
  delete reinterpret_cast<Trace*>(trace);
}

int kineto_c_trace_save(KinetoCTrace* trace, const char* path) {
  auto* t = reinterpret_cast<Trace*>(trace);
  if (t == nullptr || t->trace == nullptr || path == nullptr) {
    return 0;
  }
  t->trace->save(path);
  return 1;
}

size_t kineto_c_trace_activity_count(const KinetoCTrace* trace) {
  const auto* t = reinterpret_cast<const Trace*>(trace);
  return t != nullptr ? t->activities.size() : 0;
}

const KinetoCActivity* kineto_c_trace_activity(
    const KinetoCTrace* trace,
    size_t i) {
  const auto* t = reinterpret_cast<const Trace*>(trace);
  if (t == nullptr || i >= t->activities.size()) {
    return nullptr;
  }
  return reinterpret_cast<const KinetoCActivity*>(t->activities[i]);
}

// --- Reading an activity ----------------------------------------------------

// ITraceActivity::name() and ::metadataJson() both return `const std::string`
// -- by VALUE. Several activity kinds build the string on the fly, so there is
// nothing inside the activity to point at and c_str() on the returned temporary
// dangles the moment the call ends. Each is held in a slot here instead, which
// is enough because a caller reads one before asking for the next.
const char* kineto_c_activity_name(const KinetoCActivity* a) {
  if (a == nullptr) {
    return "";
  }
  thread_local std::string held;
  held = asActivity(a)->name();
  return held.c_str();
}

const char* kineto_c_activity_metadata_json(const KinetoCActivity* a) {
  if (a == nullptr) {
    return "";
  }
  thread_local std::string held;
  held = asActivity(a)->metadataJson();
  return held.c_str();
}

int64_t kineto_c_activity_timestamp(const KinetoCActivity* a) {
  return a != nullptr ? asActivity(a)->timestamp() : 0;
}

int64_t kineto_c_activity_duration(const KinetoCActivity* a) {
  return a != nullptr ? asActivity(a)->duration() : 0;
}

int64_t kineto_c_activity_correlation_id(const KinetoCActivity* a) {
  return a != nullptr ? asActivity(a)->correlationId() : 0;
}

int64_t kineto_c_activity_device_id(const KinetoCActivity* a) {
  return a != nullptr ? asActivity(a)->deviceId() : 0;
}

int64_t kineto_c_activity_resource_id(const KinetoCActivity* a) {
  return a != nullptr ? asActivity(a)->resourceId() : 0;
}

int32_t kineto_c_activity_type(const KinetoCActivity* a) {
  return a != nullptr ? static_cast<int32_t>(asActivity(a)->type()) : 0;
}

uint32_t kineto_c_activity_flow_id(const KinetoCActivity* a) {
  return a != nullptr ? asActivity(a)->flowId() : 0;
}

uint32_t kineto_c_activity_flow_type(const KinetoCActivity* a) {
  return a != nullptr ? asActivity(a)->flowType() : 0;
}

int kineto_c_activity_flow_start(const KinetoCActivity* a) {
  return a != nullptr && asActivity(a)->flowStart() ? 1 : 0;
}

const KinetoCActivity* kineto_c_activity_linked(const KinetoCActivity* a) {
  if (a == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<const KinetoCActivity*>(
      asActivity(a)->linkedActivity());
}

// --- Producing activities ---------------------------------------------------

KinetoCCpuTrace* kineto_c_cpu_trace_create(
    int64_t start_time,
    const char* name) {
  auto buf = std::make_unique<libkineto::CpuTraceBuffer>();
  buf->span.startTime = start_time;
  buf->span.name = name != nullptr ? name : "";
  // -1 means "not counted"; libkineto fills in the real count when it merges.
  buf->gpuOpCount = -1;
  return reinterpret_cast<KinetoCCpuTrace*>(buf.release());
}

KinetoCActivity* kineto_c_cpu_trace_add(
    KinetoCCpuTrace* buf,
    const char* name,
    int32_t type,
    int64_t device,
    int64_t resource,
    uint64_t correlation_id,
    int64_t start_time,
    int64_t end_time) {
  auto* trace = reinterpret_cast<libkineto::CpuTraceBuffer*>(buf);
  if (trace == nullptr) {
    return nullptr;
  }
  const auto activity_type = static_cast<libkineto::ActivityType>(type);
  trace->emplace_activity(
      trace->span, activity_type, name != nullptr ? name : "");
  auto& act = libkineto::CpuTraceBuffer::toRef(trace->activities.back());
  act.device = device;
  act.resource = resource;
  act.id = static_cast<int32_t>(correlation_id);
  act.startTime = start_time;
  if (activity_type != libkineto::ActivityType::CPU_INSTANT_EVENT) {
    act.endTime = end_time;
  }
  return reinterpret_cast<KinetoCActivity*>(trace->activities.back().get());
}

void kineto_c_cpu_trace_transfer(KinetoCCpuTrace* buf, int64_t end_time) {
  std::unique_ptr<libkineto::CpuTraceBuffer> trace(
      reinterpret_cast<libkineto::CpuTraceBuffer*>(buf));
  if (trace == nullptr) {
    return;
  }
  trace->span.endTime = end_time;
  libkineto::api().activityProfiler().transferCpuTrace(std::move(trace));
}

void kineto_c_activity_add_metadata(
    KinetoCActivity* a,
    const char* key,
    const char* value) {
  if (a != nullptr && key != nullptr && value != nullptr) {
    asGeneric(a)->addMetadata(key, value);
  }
}

void kineto_c_activity_set_flow(
    KinetoCActivity* a,
    uint32_t id,
    uint32_t type,
    int start) {
  if (a == nullptr) {
    return;
  }
  auto* act = asGeneric(a);
  act->flow.id = id;
  act->flow.type = type;
  act->flow.start = start != 0;
}

// --- Activity types ---------------------------------------------------------

uint32_t kineto_c_link_fwd_bwd(void) {
  return libkineto::kLinkFwdBwd;
}

} // extern "C"
