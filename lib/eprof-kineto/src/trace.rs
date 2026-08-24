
use std::ffi::{c_void, CString};

use eprof_storage::event::{Event, EventArray, EventType};
use crate::fwdbwd::{link as link_fwd_bwd, FwdBwdInput};

use super::types::{act, device_type_from_activity, CPU_TYPES, CUDA_TYPES};
use super::{
    eprof_privateuse1_backend_registered, kineto_c_activity_add_metadata,
    kineto_c_activity_correlation_id, kineto_c_activity_device_id,
    kineto_c_activity_duration, kineto_c_activity_flow_id, kineto_c_activity_flow_start,
    kineto_c_activity_flow_type, kineto_c_activity_linked, kineto_c_activity_metadata_json,
    kineto_c_activity_name, kineto_c_activity_resource_id, kineto_c_activity_set_flow,
    kineto_c_activity_timestamp, kineto_c_activity_type, kineto_c_cpu_trace_add,
    kineto_c_cpu_trace_create, kineto_c_cpu_trace_transfer, kineto_c_has_collectives_profiler,
    kineto_c_link_fwd_bwd, kineto_c_pop_correlation_id, kineto_c_pop_user_correlation_id,
    kineto_c_prepare_trace, kineto_c_push_correlation_id, kineto_c_push_user_correlation_id,
    kineto_c_record_thread_info, kineto_c_process_id, kineto_c_set_time_converter,
    kineto_c_start_trace, kineto_c_stop_trace, kineto_c_system_thread_id,
    kineto_c_time_since_epoch_now, kineto_c_toggle_collection_dynamic,
    kineto_c_trace_activity, kineto_c_trace_activity_count, read, KinetoCActivity,
    KinetoCTrace,
};

/// Now, on libkineto's clock.
pub fn now_ns() -> i64 {
    unsafe { kineto_c_time_since_epoch_now() }
}

/// The ids libkineto places a thread's activities by.
pub fn ids() -> (i32, i32) {
    unsafe {
        (
            kineto_c_process_id() as i32,
            kineto_c_system_thread_id() as i32,
        )
    }
}

pub fn record_thread_info() {
    unsafe { kineto_c_record_thread_info() };
}

pub fn push_correlation_id(id: u64, user_scope: bool) {
    unsafe {
        if user_scope {
            kineto_c_push_user_correlation_id(id);
        } else {
            kineto_c_push_correlation_id(id);
        }
    }
}

pub fn pop_correlation_id(user_scope: bool) {
    unsafe {
        if user_scope {
            kineto_c_pop_user_correlation_id();
        } else {
            kineto_c_pop_correlation_id();
        }
    }
}

pub unsafe fn set_time_converter(
    convert: Option<extern "C" fn(*mut c_void, i64) -> i64>,
    ctx: *mut c_void,
) {
    kineto_c_set_time_converter(convert, ctx);
}

/// Hands our events over, ends the trace and merges what kineto collected.
pub unsafe fn finish(arr: *mut EventArray, start_time_ns: i64, end_time_ns: i64)
    -> *mut super::KinetoCTrace {
    pass_events_to_kineto(&*arr, start_time_ns, end_time_ns);
    stop_and_transfer(&mut *arr)
}

// --- Lifecycle --------------------------------------------------------------

/// Registers, initializes and prepares a trace.
pub fn prepare_trace(cpu_only: bool, want_cpu: bool, want_cuda: bool, trace_id: &str) {
    let mut types: Vec<i32> = Vec::new();
    if want_cpu {
        types.extend_from_slice(CPU_TYPES);
    }
    if want_cuda {
        types.extend_from_slice(CUDA_TYPES);
    }
    if unsafe { kineto_c_has_collectives_profiler() } != 0 {
        types.push(act::COLLECTIVE_COMM);
    }
    types.sort_unstable();
    types.dedup();

    let config = if trace_id.is_empty() {
        String::new()
    } else {
        format!("REQUEST_TRACE_ID={trace_id}\nREQUEST_GROUP_TRACE_ID={trace_id}\n")
    };
    let config = CString::new(config).unwrap_or_default();
    unsafe {
        kineto_c_prepare_trace(
            cpu_only as i32,
            types.as_ptr(),
            types.len(),
            config.as_ptr(),
        );
    }
}

// --- Handing our events to kineto -------------------------------------------

const INDEX_KEY: &str = "Ev Idx";

/// Reads the index back out of an activity's metadata JSON.
fn extract_index(metadata_json: &str) -> Option<usize> {
    let prefix = format!("\"{INDEX_KEY}\": ");
    let at = metadata_json.find(&prefix)? + prefix.len();
    let digits: String = metadata_json[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn generate_forward_backward_links(activities: &[*mut KinetoCActivity], arr: &EventArray) {
    // Candidates are the torch ops carrying a sequence number, in end-time
    // order -- the matcher depends on that ordering.
    let mut candidates: Vec<usize> = (0..arr.len())
        .filter(|&i| {
            arr.get(i).is_some_and(|e| {
                e.tag() == EventType::TorchOp && e.sequence_number >= 0
            })
        })
        .collect();
    candidates.sort_by_key(|&i| arr.end_time_ns(i));

    let inputs: Vec<FwdBwdInput> = candidates
        .iter()
        .map(|&i| {
            let e = arr.get(i).expect("candidate index came from this array");
            FwdBwdInput {
                forward_tid: e.forward_tid,
                sequence_number: e.sequence_number,
                start_tid: e.start_tid,
                start_time: e.start_time_ns,
            }
        })
        .collect();

    let links = link_fwd_bwd(&inputs, /*first_flow_id=*/1);

    let fwd_bwd = unsafe { kineto_c_link_fwd_bwd() };
    for link in &links {
        let fwd = candidates[link.forward_idx as usize];
        let bwd = candidates[link.backward_idx as usize];
        unsafe {
            kineto_c_activity_set_flow(activities[fwd], link.flow_id, fwd_bwd, 1);
            kineto_c_activity_set_flow(activities[bwd], link.flow_id, fwd_bwd, 0);
        }
    }
}

/// Builds a libkineto activity for every event we recorded and transfers them.
pub fn pass_events_to_kineto(arr: &EventArray, start_time_ns: i64, end_time_ns: i64) {
    let name = CString::new("PyTorch Profiler").unwrap_or_default();
    let buf = unsafe { kineto_c_cpu_trace_create(start_time_ns, name.as_ptr()) };
    let index_key = CString::new(INDEX_KEY).unwrap_or_default();

    let mut activities: Vec<*mut KinetoCActivity> = Vec::with_capacity(arr.len());
    for i in 0..arr.len() {
        let e = arr.get(i).expect("i < len");
        let start = e.start_time_ns;
        // An async op can leave an end time of i64::MIN; clamping keeps kineto
        // from computing a duration that overflows into a huge positive number.
        let end = arr.end_time_ns(i).max(start);
        let display = CString::new(e.display_name()).unwrap_or_default();
        let activity = unsafe {
            kineto_c_cpu_trace_add(
                buf,
                display.as_ptr(),
                arr.kineto_type(i),
                e.device as i64,
                e.resource as i64,
                arr.correlation_id(i),
                start,
                end,
            )
        };
        let idx = CString::new(i.to_string()).unwrap_or_default();
        unsafe { kineto_c_activity_add_metadata(activity, index_key.as_ptr(), idx.as_ptr()) };
        activities.push(activity);
    }

    generate_forward_backward_links(&activities, arr);
    unsafe { kineto_c_cpu_trace_transfer(buf, end_time_ns) };
}

// --- Reading back what kineto collected -------------------------------------

struct Transfer<'a> {
    arr: &'a mut EventArray,
    privateuse1: bool,
    /// Activity address -> the event it is.
    seen: std::collections::HashMap<usize, i64>,
}

const UNMATCHED: i64 = -1;
/// Activities kineto produced have no thread of ours.
const NO_TID: u64 = u64::MAX;

impl<'a> Transfer<'a> {
    fn lookup(&mut self, activity: *const KinetoCActivity) -> i64 {
        if activity.is_null() {
            return UNMATCHED;
        }
        if let Some(&i) = self.seen.get(&(activity as usize)) {
            return i;
        }
        let json = unsafe { read(kineto_c_activity_metadata_json(activity)) };
        match extract_index(&json) {
            Some(i) if i < self.arr.len() => {
                self.seen.insert(activity as usize, i as i64);
                i as i64
            }
            _ => UNMATCHED,
        }
    }

    fn reassociate(&mut self, activities: &[*const KinetoCActivity]) {
        for &activity in activities {
            let i = self.lookup(activity);
            if i != UNMATCHED {
                if let Some(e) = self.arr.get_mut(i as usize) {
                    e.activity_ptr = activity as u64;
                }
            }
        }
    }

    fn event_from_activity(&mut self, activity: *const KinetoCActivity) -> i64 {
        let activity_type = unsafe { kineto_c_activity_type(activity) };
        let i = self.arr.push(Event {
            tag_raw: EventType::Kineto as i32,
            start_time_ns: unsafe { kineto_c_activity_timestamp(activity) },
            start_tid: NO_TID,
            device: unsafe { kineto_c_activity_device_id(activity) } as i32,
            resource: unsafe { kineto_c_activity_resource_id(activity) } as i32,
            // A kineto event carries its duration, not its end: end_time_ns()
            // derives the end from it.
            duration_ns: unsafe { kineto_c_activity_duration(activity) },
            correlation_id: unsafe { kineto_c_activity_correlation_id(activity) } as u64,
            activity_type,
            flow_id: unsafe { kineto_c_activity_flow_id(activity) },
            flow_type: unsafe { kineto_c_activity_flow_type(activity) },
            flow_start: unsafe { kineto_c_activity_flow_start(activity) } as u32,
            device_type: device_type_from_activity(activity_type, self.privateuse1),
            name: unsafe { read(kineto_c_activity_name(activity)) },
            parent: -1,
            linked: -1,
            ..Default::default()
        });
        i as i64
    }

    fn to_event(&mut self, activity: *const KinetoCActivity) -> i64 {
        let i = self.lookup(activity);
        if i != UNMATCHED {
            return i;
        }
        let activity_type = unsafe { kineto_c_activity_type(activity) };
        if matches!(
            activity_type,
            act::CPU_OP | act::CPU_INSTANT_EVENT | act::USER_ANNOTATION | act::PYTHON_FUNCTION
        ) {
            eprintln!(
                "[eprof] Detected an event which was likely passed to kineto by the \
                 profiler, but is not present in the set of known events: {}. This most \
                 likely means kineto has not maintained address stability for it.",
                unsafe { read(kineto_c_activity_name(activity)) }
            );
            return UNMATCHED;
        }
        let i = self.event_from_activity(activity);
        self.seen.insert(activity as usize, i);
        i
    }

    fn extract(&mut self, activities: &[*const KinetoCActivity]) {
        for &activity in activities {
            let i = self.to_event(activity);
            let linked = unsafe { kineto_c_activity_linked(activity) };
            if i != UNMATCHED && !linked.is_null() {
                let to = self.to_event(linked);
                if let Some(e) = self.arr.get_mut(i as usize) {
                    e.linked = to;
                }
            }
        }
    }
}

/// Ends the trace and merges what kineto collected into `arr`.
pub fn stop_and_transfer(arr: &mut EventArray) -> *mut KinetoCTrace {
    let trace = unsafe { kineto_c_stop_trace() };
    if trace.is_null() {
        return trace;
    }
    let n = unsafe { kineto_c_trace_activity_count(trace) };
    let activities: Vec<*const KinetoCActivity> = (0..n)
        .map(|i| unsafe { kineto_c_trace_activity(trace, i) })
        .filter(|p| !p.is_null())
        .collect();

    let privateuse1 = unsafe { eprof_privateuse1_backend_registered() } != 0;
    let mut transfer = Transfer {
        arr,
        privateuse1,
        seen: std::collections::HashMap::new(),
    };
    transfer.reassociate(&activities);
    transfer.extract(&activities);
    trace
}

pub fn attach_metadata(arr: &mut EventArray) {
    for i in 0..arr.len() {
        let activity = arr.get(i).map_or(0, |e| e.activity_ptr) as *mut KinetoCActivity;
        if !activity.is_null() {
            for (key, value) in arr.kineto_metadata(i) {
                let (k, v) = (
                    CString::new(key).unwrap_or_default(),
                    CString::new(value).unwrap_or_default(),
                );
                unsafe { kineto_c_activity_add_metadata(activity, k.as_ptr(), v.as_ptr()) };
            }
        }
        if let Some(e) = arr.get_mut(i) {
            e.activity_ptr = 0;
        }
    }
}

// --- The rest of the trace lifecycle ----------------------------------------

pub fn start_trace() {
    unsafe { kineto_c_start_trace() };
}

pub fn toggle(enable: i32) {
    unsafe { kineto_c_toggle_collection_dynamic(enable) };
}

pub unsafe fn eprof_kineto_finish(
    arr: *mut EventArray,
    start_time_ns: i64,
    end_time_ns: i64,
) -> *mut KinetoCTrace {
    if arr.is_null() {
        return std::ptr::null_mut();
    }
    pass_events_to_kineto(&*arr, start_time_ns, end_time_ns);
    stop_and_transfer(&mut *arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_index_is_read_back_out_of_the_metadata_json() {
        assert_eq!(extract_index(r#"{"Ev Idx": 42, "x": 1}"#), Some(42));
        assert_eq!(extract_index(r#"{"x": 1, "Ev Idx": 7}"#), Some(7));
    }

    #[test]
    fn an_activity_without_our_index_is_not_ours() {
        // This is how an activity kineto produced itself is recognised.
        assert_eq!(extract_index(r#"{"stream": 7, "correlation": 3}"#), None);
        assert_eq!(extract_index(""), None);
    }
}
