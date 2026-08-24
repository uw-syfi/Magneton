/* The queue an active profiler collects into (eprof-torch/src/queue.rs).
 *
 * A subqueue exists per recording thread so the hot path never contends; the
 * queue merges them at the end. C++ reads the at::RecordFunction and the
 * c10::IValues and pushes the results through here.
 */

#ifndef EPROF_RUST_QUEUE_H
#define EPROF_RUST_QUEUE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EprofRecordQueue EprofRecordQueue;
typedef struct EprofSubqueue EprofSubqueue;

/* Adds to the metadata of the op most recently begun. `is_kwinput` selects
 * between the keyword arguments and the collective-communication metadata. */
void eprof_subqueue_add_op_metadata(EprofSubqueue *sq, const char *key,
                                    const char *value, int32_t is_kwinput);

/* The gathered python frame entries, for the tracer's replay. The pointer is
 * valid until the queue is destroyed. */
void *eprof_queue_python_enters(EprofRecordQueue *queue, size_t *out_len);

/* Appends an entry that did not come from a subqueue: a frame already on the
 * stack when profiling started, which has no thread of its own on record. */
void eprof_queue_push_python_enter(EprofRecordQueue *queue, uint64_t key,
                                   uint64_t system_tid, int64_t start_ns);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EPROF_RUST_QUEUE_H */
