/* A profiling run (eprof-torch/src/run.rs).
 *
 * These are the calls made from inside a torch callback, while the workload is
 * running. Each is coarse on purpose: the caller hands over what it just read
 * and takes no decisions, so nothing on this path costs more than an append.
 */

#ifndef EPROF_RUST_RUN_H
#define EPROF_RUST_RUN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Records an op's start; returns the index the exit hands back. `fn` is a
 * const at::RecordFunction*, valid for this call only -- Rust passes it back
 * to the accessors in include/capture.h and never dereferences it. */
size_t eprof_begin_op(void *queue, const void *fn);

/* Completes the op `eprof_begin_op` returned `index` for. */
void eprof_end_op(void *queue, size_t index, int32_t user_scope);

/* Records a python frame entry on the calling thread's subqueue. */
void eprof_queue_push_py_call(void *queue, uint64_t key, int64_t start_time);

/* Whether this run records input shapes; answers `needsInputs`. */
int32_t eprof_run_records_shapes(void);

/* The two allocation reports torch makes on the profiler state. */
void eprof_queue_report_alloc(void *queue, uint64_t ptr, int64_t alloc_size,
                              size_t total_allocated, size_t total_reserved,
                              int8_t device_type, int8_t device_index);
void eprof_queue_report_oom(void *queue, int64_t alloc_size,
                            size_t total_allocated, size_t total_reserved,
                            int8_t device_type, int8_t device_index);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EPROF_RUST_RUN_H */
