//! Driving libkineto.

use std::ffi::{c_char, c_void, CStr};

pub mod fwdbwd;
pub mod result;
pub mod trace;
pub mod types;

pub use result::*;
pub use trace::*;
pub use types::*;

// --- The C boundary ---------------------------------------------------------

#[repr(C)]
pub struct KinetoCTrace {
    _private: [u8; 0],
}
#[repr(C)]
pub struct KinetoCCpuTrace {
    _private: [u8; 0],
}
#[repr(C)]
pub struct KinetoCActivity {
    _private: [u8; 0],
}

extern "C" {
    fn kineto_c_prepare_trace(
        cpu_only: i32,
        activity_types: *const i32,
        n_activity_types: usize,
        config: *const c_char,
    );
    fn kineto_c_start_trace();
    fn kineto_c_stop_trace() -> *mut KinetoCTrace;
    fn kineto_c_toggle_collection_dynamic(enable: i32);
    fn kineto_c_push_correlation_id(id: u64);
    fn kineto_c_pop_correlation_id();
    fn kineto_c_push_user_correlation_id(id: u64);
    fn kineto_c_pop_user_correlation_id();
    fn kineto_c_record_thread_info();
    fn kineto_c_process_id() -> i64;
    fn kineto_c_system_thread_id() -> i64;
    fn kineto_c_time_since_epoch_now() -> i64;
    fn kineto_c_has_collectives_profiler() -> i32;
    fn kineto_c_set_time_converter(
        convert: Option<extern "C" fn(*mut c_void, i64) -> i64>,
        ctx: *mut c_void,
    );
    fn kineto_c_trace_free(trace: *mut KinetoCTrace);
    fn kineto_c_trace_save(trace: *mut KinetoCTrace, path: *const c_char) -> i32;
    fn kineto_c_trace_activity_count(trace: *const KinetoCTrace) -> usize;
    fn kineto_c_trace_activity(trace: *const KinetoCTrace, i: usize)
        -> *const KinetoCActivity;
    fn kineto_c_activity_name(a: *const KinetoCActivity) -> *const c_char;
    fn kineto_c_activity_metadata_json(a: *const KinetoCActivity) -> *const c_char;
    fn kineto_c_activity_timestamp(a: *const KinetoCActivity) -> i64;
    fn kineto_c_activity_duration(a: *const KinetoCActivity) -> i64;
    fn kineto_c_activity_correlation_id(a: *const KinetoCActivity) -> i64;
    fn kineto_c_activity_device_id(a: *const KinetoCActivity) -> i64;
    fn kineto_c_activity_resource_id(a: *const KinetoCActivity) -> i64;
    fn kineto_c_activity_type(a: *const KinetoCActivity) -> i32;
    fn kineto_c_activity_flow_id(a: *const KinetoCActivity) -> u32;
    fn kineto_c_activity_flow_type(a: *const KinetoCActivity) -> u32;
    fn kineto_c_activity_flow_start(a: *const KinetoCActivity) -> i32;
    fn kineto_c_activity_linked(a: *const KinetoCActivity) -> *const KinetoCActivity;
    fn kineto_c_cpu_trace_create(start_time: i64, name: *const c_char)
        -> *mut KinetoCCpuTrace;
    #[allow(clippy::too_many_arguments)]
    fn kineto_c_cpu_trace_add(
        buf: *mut KinetoCCpuTrace,
        name: *const c_char,
        activity_type: i32,
        device: i64,
        resource: i64,
        correlation_id: u64,
        start_time: i64,
        end_time: i64,
    ) -> *mut KinetoCActivity;
    fn kineto_c_cpu_trace_transfer(buf: *mut KinetoCCpuTrace, end_time: i64);
    fn kineto_c_activity_add_metadata(
        a: *mut KinetoCActivity,
        key: *const c_char,
        value: *const c_char,
    );
    fn kineto_c_activity_set_flow(a: *mut KinetoCActivity, id: u32, ty: u32, start: i32);
    fn kineto_c_link_fwd_bwd() -> u32;
}

extern "C" {
    fn eprof_privateuse1_backend_registered() -> i32;
}

pub(crate) unsafe fn read(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
