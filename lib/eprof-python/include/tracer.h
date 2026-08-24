#pragma once


#include <stdint.h>

extern "C" {

/* The CPython tracer, as a handle. It is C++ because CPython is: PyEval_SetProfile,
 * the TraceContext PyTypeObject, frame walking and the GIL. Everything it records
 * goes straight into the Rust caches it was given.
 *
 * `queue` is an EprofRecordQueue*. Returns null when the tracer cannot start,
 * which the caller treats as "no python frames this run". */
void *eprof_tracer_create(void *queue);

/* Stops recording. Safe to call more than once. */
void eprof_tracer_stop(void *tracer);

/* Replays the recorded frames onto `events`. The entries recorded during
 * collection are already on `queue`; this adds the frames that were on the
 * stack before profiling began, then replays the lot. `convert` turns a raw
 * clock reading into trace time. */
void eprof_tracer_get_events(void *tracer, void *events, void *queue,
                             int64_t (*convert)(void *ctx, int64_t t), void *ctx,
                             int64_t end_time_ns);

void eprof_tracer_destroy(void *tracer);

/* Registers the tracer's python type. Call once when the module loads. */
void eprof_tracer_init(void);

}  /* extern "C" */
