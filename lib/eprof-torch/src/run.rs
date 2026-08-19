//! A profiling run: what it collects into, and how it is wound down.
//!
//! Everything about a run lives here. C++ holds one object for it -- the
//! `ThreadLocalState` shell torch reports allocations into -- and that shell
//! holds nothing but a pointer back to this.
//!
//! The two RecordFunction callbacks land in `begin_op` / `end_op` below. They
//! arrive with a `const at::RecordFunction*` this module never dereferences:
//! it hands the pointer back to the coarse accessors in `include/capture.h`,
//! which are the reads that need libtorch types to perform.

use std::ffi::{c_char, c_void};

use crate::inputs::OpInputs;
use crate::pods::{PodAlloc, PodOom};
use crate::queue::{RecordQueue, Subqueue};
use eprof_storage::event::EventArray;
use eprof_kineto::{self, KinetoCTrace};

/// Everything read off one `at::RecordFunction`, in a single crossing.
/// Laid out to match `EprofOpFields` in include/capture.h, which fills it.
#[repr(C)]
#[derive(Default)]
struct OpFields {
    name: *const c_char,
    sequence_number: i64,
    forward_tid: u64,
    scope: u8,
    record_function_id: u64,
    is_nccl_meta: i32,
    is_user_scope: i32,
}

// OpInputs and Subqueue cross as opaque handles; C++ only passes them back.
#[allow(improper_ctypes)]
extern "C" {
    fn eprof_rf_read(fn_: *const c_void, out: *mut OpFields);
    fn eprof_rf_push_inputs(fn_: *const c_void, inputs: *mut OpInputs);
    fn eprof_rf_push_kwinputs(fn_: *const c_void, subqueue: *mut Subqueue);
    fn eprof_rf_push_nccl_meta(fn_: *const c_void, subqueue: *mut Subqueue);

    /// Creates the shell torch reports into and pushes it onto c10's stack.
    fn eprof_state_push(queue: *mut RecordQueue, profile_memory: i32) -> *mut c_void;
    /// Pops it, and installs the run's clock conversion for `eprof_convert_time`.
    fn eprof_state_pop();
    fn eprof_convert_time(ctx: *mut c_void, t: i64) -> i64;
    fn eprof_callbacks_push();
    fn eprof_callbacks_remove();
    fn eprof_current_thread_id() -> u64;
    /// c10's raw clock reading, which the run's converter is calibrated
    /// against. One call per op stamp; taking it anywhere else would not
    /// convert correctly.
    fn eprof_approx_now() -> i64;
    fn eprof_device_synchronize();

    fn eprof_tracer_create(queue: *mut RecordQueue) -> *mut c_void;
    fn eprof_tracer_stop(tracer: *mut c_void);
    fn eprof_tracer_get_events(
        tracer: *mut c_void,
        events: *mut EventArray,
        queue: *mut RecordQueue,
        convert: extern "C" fn(*mut c_void, i64) -> i64,
        ctx: *mut c_void,
        end_time_ns: i64,
    );
    fn eprof_tracer_destroy(tracer: *mut c_void);
}

/// Trampoline: the installed conversion, as a plain fn pointer to hand around.
extern "C" fn convert_time(ctx: *mut c_void, t: i64) -> i64 {
    unsafe { eprof_convert_time(ctx, t) }
}

/// Everything a run owns.
pub struct Run {
    pub start_time_ns: i64,
    pub record_shapes: bool,
    pub events: *mut EventArray,
    pub queue: *mut RecordQueue,
    /// The CPython tracer, or null when this run does not trace python.
    tracer: *mut c_void,
}

impl Run {
    /// Creates the stores, the queue, and the tracer if one was asked for, then
    /// puts the shell torch reports into on c10's stack.
    ///
    /// # Safety
    /// One run at a time, on the thread that will call `finish`.
    pub unsafe fn start(record_shapes: bool, profile_memory: bool, trace_python: bool) -> Run {
        let events = Box::into_raw(Box::new(EventArray::default()));
        let queue = crate::queue::create(events);
        let tracer = if trace_python {
            eprof_tracer_create(queue)
        } else {
            std::ptr::null_mut()
        };
        let mut run = Run {
            start_time_ns: eprof_kineto::now_ns(),
            record_shapes,
            events,
            queue,
            tracer,
        };
        eprof_state_push(queue, profile_memory as i32);
        run.start_time_ns = eprof_kineto::now_ns();
        run
    }

    /// Merges everything collected, ends the kineto trace and returns it. The
    /// event array passes to the result; the queue is dropped here.
    ///
    /// # Safety
    /// The run must have been started and not already finished.
    pub unsafe fn finish(&mut self) -> *mut KinetoCTrace {
        let end_time_ns = eprof_kineto::now_ns();
        if !self.tracer.is_null() {
            eprof_tracer_stop(self.tracer);
        }

        // Popping the state is what builds the run's clock conversion, so it
        // has to happen before anything converts a timestamp.
        eprof_state_pop();
        eprof_kineto::set_time_converter(Some(convert_time), std::ptr::null_mut());

        crate::queue::merge(self.queue, convert_time, std::ptr::null_mut());

        if !self.tracer.is_null() {
            eprof_tracer_get_events(
                self.tracer,
                self.events,
                self.queue,
                convert_time,
                std::ptr::null_mut(),
                end_time_ns,
            );
            eprof_tracer_destroy(self.tracer);
            self.tracer = std::ptr::null_mut();
        }

        // Hands our events to kineto, ends the trace and merges what kineto
        // collected back in -- one call, because the transfer has to happen
        // before the stop.
        let trace = eprof_kineto::finish(self.events, self.start_time_ns, end_time_ns);

        // Order by start time, parent, then attach metadata: the call stack and
        // python parent id are read off the tree, so they need it to exist.
        (*self.events).sort_by_start_time();
        (*self.events).materialize(eprof_current_thread_id());
        eprof_kineto::attach_metadata(&mut *self.events);

        crate::queue::destroy(self.queue);
        self.queue = std::ptr::null_mut();
        trace
    }
}

// --- The RecordFunction callbacks -------------------------------------------

/// Records an op's start and returns its index, which the exit callback hands
/// back. Called from `OnFunctionEnter` with the live RecordFunction.
///
/// # Safety
/// `queue` must be a live queue; `fn_` a `const at::RecordFunction*` valid for
/// this call.
#[no_mangle]
pub unsafe extern "C" fn eprof_begin_op(queue: *mut c_void, fn_: *const c_void) -> usize {
    let queue = queue.cast::<RecordQueue>();
    if queue.is_null() {
        return 0;
    }
    let mut f = OpFields::default();
    eprof_rf_read(fn_, &mut f);

    let sq = crate::queue::subqueue_for_current_thread(queue);
    let name = if f.name.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(f.name).to_string_lossy().into_owned()
    };

    let (index, correlation_id) = (*sq).ops.push(
        &name,
        f.sequence_number,
        f.forward_tid,
        f.scope,
        f.record_function_id,
        0,
    );

    // One metadata slot per op either way: materialization reads these
    // positionally alongside the op list.
    (*sq).extra_meta.push(Vec::new());
    (*sq).kwinputs.push(Vec::new());

    if RECORD_SHAPES.with(|r| r.get()) {
        eprof_rf_push_inputs(fn_, &mut (*sq).inputs);
        eprof_rf_push_kwinputs(fn_, sq);
    }
    if f.is_nccl_meta != 0 {
        eprof_rf_push_nccl_meta(fn_, sq);
    }

    eprof_kineto::push_correlation_id(correlation_id, f.is_user_scope != 0);

    // Stamped last, so the op is not charged for the bookkeeping above.
    (*sq).ops.set_start(index, eprof_approx_now());
    index
}

/// Completes the op `begin_op` returned `index` for.
///
/// # Safety
/// `queue` must be the same queue the op was begun on.
#[no_mangle]
pub unsafe extern "C" fn eprof_end_op(queue: *mut c_void, index: usize, user_scope: i32) {
    let queue = queue.cast::<RecordQueue>();
    if queue.is_null() {
        return;
    }
    let sq = crate::queue::subqueue_for_current_thread(queue);
    (*sq).ops.set_end(index, eprof_approx_now(), eprof_current_thread_id());
    eprof_kineto::pop_correlation_id(user_scope != 0);
}

thread_local! {
    /// Whether this run records input shapes. Read on every op, so it is kept
    /// here rather than fetched across the boundary.
    static RECORD_SHAPES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Answers `needsInputs` when the callbacks are registered.
#[no_mangle]
pub extern "C" fn eprof_run_records_shapes() -> i32 {
    RECORD_SHAPES.with(|r| r.get()) as i32
}

pub fn set_record_shapes(on: bool) {
    RECORD_SHAPES.with(|r| r.set(on));
}

/// Records a python frame entry on the calling thread's subqueue.
///
/// # Safety
/// `queue` must be a live queue.
#[no_mangle]
pub unsafe extern "C" fn eprof_queue_push_py_call(
    queue: *mut c_void,
    key: u64,
    start_time: i64,
) {
    let queue = queue.cast::<RecordQueue>();
    if queue.is_null() {
        return;
    }
    let sq = crate::queue::subqueue_for_current_thread(queue);
    (*sq).py_calls.push((key, start_time));
}

// --- Allocation reports -----------------------------------------------------

/// # Safety
/// `queue` must be a live queue.
#[no_mangle]
pub unsafe extern "C" fn eprof_queue_report_alloc(
    queue: *mut RecordQueue,
    ptr: u64,
    alloc_size: i64,
    total_allocated: usize,
    total_reserved: usize,
    device_type: i8,
    device_index: i8,
) {
    if queue.is_null() {
        return;
    }
    let sq = crate::queue::subqueue_for_current_thread(queue);
    (*sq).pods.push_alloc(PodAlloc {
        start_time: eprof_approx_now(),
        ptr,
        alloc_size,
        total_allocated: total_allocated as u64,
        total_reserved: total_reserved as u64,
        device_type,
        device_index,
    });
}

/// # Safety
/// `queue` must be a live queue.
#[no_mangle]
pub unsafe extern "C" fn eprof_queue_report_oom(
    queue: *mut RecordQueue,
    alloc_size: i64,
    total_allocated: usize,
    total_reserved: usize,
    device_type: i8,
    device_index: i8,
) {
    if queue.is_null() {
        return;
    }
    let sq = crate::queue::subqueue_for_current_thread(queue);
    (*sq).pods.push_oom(PodOom {
        start_time: eprof_approx_now(),
        alloc_size,
        total_allocated: total_allocated as u64,
        total_reserved: total_reserved as u64,
        device_type,
        device_index,
    });
}

/// Re-exported so the driver can register the callbacks and tear them down.
///
/// # Safety
/// A run must be in progress.
pub unsafe fn callbacks_push() {
    eprof_callbacks_push();
}

/// # Safety
/// A run must be in progress.
pub unsafe fn callbacks_remove() {
    eprof_callbacks_remove();
}

/// # Safety
/// Only meaningful with CUDA activity requested.
pub unsafe fn device_synchronize() {
    eprof_device_synchronize();
}

/// Whether a run's state is on this thread's c10 stack.
pub fn state_active() -> bool {
    extern "C" {
        fn eprof_state_active() -> i32;
    }
    unsafe { eprof_state_active() != 0 }
}
