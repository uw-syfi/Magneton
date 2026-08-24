/*
 * A C ABI over the parts of libkineto a profiler actually drives.
 *
 * libkineto's own interface is C++ -- virtual activity classes, a deque of
 * unique_ptrs, std::function hooks -- so a host written in anything else has
 * to keep a C++ translation layer of its own just to reach it. This is that
 * layer, kept here rather than in each host, so the host only has to speak C.
 *
 * Everything here is a thin forward to libkineto::api(); no policy lives in
 * this file. Which activity types to record, how to correlate them and what
 * an activity means are the caller's business.
 *
 * Threading follows libkineto: the trace lifecycle calls are expected on the
 * controlling thread, while the correlation-id push/pop pair is thread-local
 * and belongs on whichever thread is recording.
 */

#ifndef KINETO_C_API_H
#define KINETO_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. A trace comes from kineto_c_stop_trace and must be freed
 * with kineto_c_trace_free; activities belong to whatever produced them and
 * stay valid until it goes away. */
typedef struct KinetoCTrace KinetoCTrace;
typedef struct KinetoCCpuTrace KinetoCCpuTrace;
typedef struct KinetoCActivity KinetoCActivity;

/* --- Lifecycle ---------------------------------------------------------- */

/* Registers and initializes the profiler if that has not happened yet, then
 * prepares a trace for the given libkineto::ActivityType values. `config` is
 * libkineto's newline-separated config text, or NULL for none. */
void kineto_c_prepare_trace(
    int cpu_only,
    const int32_t* activity_types,
    size_t n_activity_types,
    const char* config);

void kineto_c_start_trace(void);

/* Ends collection and returns the trace, or NULL if libkineto produced none. */
KinetoCTrace* kineto_c_stop_trace(void);

void kineto_c_toggle_collection_dynamic(int enable);

/* Correlation ids tie a recorded op to the kernels it launches. Thread-local:
 * push before the work, pop after, on the thread doing the work. */
void kineto_c_push_correlation_id(uint64_t id);
void kineto_c_pop_correlation_id(void);
void kineto_c_push_user_correlation_id(uint64_t id);
void kineto_c_pop_user_correlation_id(void);

/* Registers the calling thread so its activities are labelled in the trace. */
void kineto_c_record_thread_info(void);

/* The ids libkineto itself uses to place an activity: the process, and the
 * calling thread. */
int64_t kineto_c_process_id(void);
int64_t kineto_c_system_thread_id(void);

/* Now, on libkineto's clock. */
int64_t kineto_c_time_since_epoch_now(void);

/* Whether this build has the collective-communication profiler compiled in,
 * i.e. whether COLLECTIVE_COMM is worth asking for. */
int kineto_c_has_collectives_profiler(void);

/* Installs the conversion from the caller's raw clock readings to trace time.
 * libkineto applies it to timestamps it did not stamp itself. Passing NULL
 * restores the identity. */
void kineto_c_set_time_converter(
    int64_t (*convert)(void* ctx, int64_t t),
    void* ctx);

/* --- Reading a finished trace ------------------------------------------- */

void kineto_c_trace_free(KinetoCTrace* trace);

/* Writes the trace to `path` as chrome-trace JSON. Returns 0 on failure. */
int kineto_c_trace_save(KinetoCTrace* trace, const char* path);

size_t kineto_c_trace_activity_count(const KinetoCTrace* trace);

/* Activity `i`, or NULL if out of range. Valid until the trace is freed. */
const KinetoCActivity* kineto_c_trace_activity(
    const KinetoCTrace* trace,
    size_t i);

/* --- Reading an activity ------------------------------------------------ */

/* Valid for as long as the activity is. */
const char* kineto_c_activity_name(const KinetoCActivity* a);
const char* kineto_c_activity_metadata_json(const KinetoCActivity* a);

int64_t kineto_c_activity_timestamp(const KinetoCActivity* a);
int64_t kineto_c_activity_duration(const KinetoCActivity* a);
int64_t kineto_c_activity_correlation_id(const KinetoCActivity* a);
int64_t kineto_c_activity_device_id(const KinetoCActivity* a);
int64_t kineto_c_activity_resource_id(const KinetoCActivity* a);

/* A libkineto::ActivityType value. */
int32_t kineto_c_activity_type(const KinetoCActivity* a);

/* Flows express the launch relationship a correlation id cannot: the runtime
 * call that launched a kernel is the flow's start, the kernel shares its id. */
uint32_t kineto_c_activity_flow_id(const KinetoCActivity* a);
uint32_t kineto_c_activity_flow_type(const KinetoCActivity* a);
int kineto_c_activity_flow_start(const KinetoCActivity* a);

/* The activity this one was correlated to, or NULL. */
const KinetoCActivity* kineto_c_activity_linked(const KinetoCActivity* a);

/* --- Producing activities ----------------------------------------------- */

/* A buffer of activities the caller recorded, to be handed to libkineto so
 * they appear in the trace alongside the ones it collected itself. */
KinetoCCpuTrace* kineto_c_cpu_trace_create(int64_t start_time, const char* name);

/* Appends one activity and returns it, still owned by the buffer. `type` is a
 * libkineto::ActivityType value; an instant event ignores `end_time`. */
KinetoCActivity* kineto_c_cpu_trace_add(
    KinetoCCpuTrace* buf,
    const char* name,
    int32_t type,
    int64_t device,
    int64_t resource,
    uint64_t correlation_id,
    int64_t start_time,
    int64_t end_time);

/* Hands the buffer to libkineto, which takes ownership; the handle and every
 * activity in it are dead afterwards. MUST happen before stop_trace: libkineto
 * merges these with what it collected while ending the trace, and a buffer
 * transferred after that point is simply dropped. */
void kineto_c_cpu_trace_transfer(KinetoCCpuTrace* buf, int64_t end_time);

/* These write to an activity the caller produced. They are not valid on one
 * that came out of a trace -- libkineto's own activities are read-only, and
 * are not necessarily the same concrete type. */
void kineto_c_activity_add_metadata(
    KinetoCActivity* a,
    const char* key,
    const char* value);
void kineto_c_activity_set_flow(
    KinetoCActivity* a,
    uint32_t id,
    uint32_t type,
    int start);

/* --- Activity types ----------------------------------------------------- */

/* The value of libkineto::kLinkFwdBwd, for tagging a forward/backward flow. */
uint32_t kineto_c_link_fwd_bwd(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KINETO_C_API_H */
