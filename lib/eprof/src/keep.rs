//! Keeps the C++ leaves' Rust entry points in the module.

/// An address, and nothing else is ever done with it.
#[repr(transparent)]
struct Entry(*const ());

// SAFETY: never dereferenced, never read. The array exists so the linker sees
// a reference to each function.
unsafe impl Sync for Entry {}

macro_rules! keep {
    ($($path:path),* $(,)?) => {
        #[used]
        static KEEP: &[Entry] = &[$(Entry($path as *const ())),*];
    };
}

keep![
    eprof_torch::run::eprof_begin_op,
    eprof_torch::run::eprof_end_op,
    eprof_torch::run::eprof_queue_push_py_call,
    eprof_torch::run::eprof_queue_report_alloc,
    eprof_torch::run::eprof_queue_report_oom,
    eprof_torch::run::eprof_run_records_shapes,
    eprof_torch::queue::eprof_queue_push_python_enter,
    eprof_torch::queue::eprof_queue_python_enters,
    eprof_torch::queue::eprof_subqueue_add_op_metadata,
    eprof_torch::inputs::eprof_op_inputs_push_scalar,
    eprof_torch::inputs::eprof_op_inputs_push_tag,
    eprof_torch::inputs::eprof_op_inputs_push_tensor,
    eprof_python::cache::eprof_pycache_create,
    eprof_python::cache::eprof_pycache_destroy,
    eprof_python::cache::eprof_pycache_get,
    eprof_python::cache::eprof_pycache_intern_site,
    eprof_python::cache::eprof_pycache_trim_prefixes,
    eprof_python::replay::eprof_pycache_replay,
];
