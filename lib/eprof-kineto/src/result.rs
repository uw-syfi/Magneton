//! What a finished run hands back.

use std::ffi::CString;
use std::path::Path;

use eprof_storage::event::EventArray;

use super::{kineto_c_trace_free, kineto_c_trace_save, KinetoCTrace};

/// What a finished run hands back: the events, and the kineto trace they were
/// merged with. Owns both, which is the only reason it exists as a type -- the
/// two are freed together and neither outlives it.
///
/// The kineto trace is a C++ object, so it is held as a raw handle and released
/// in `Drop`. The events are Rust's; the box is only so the trace and the array
/// share one lifetime.
pub struct ProfilerResult {
    pub trace_start_ns: u64,
    trace: *mut KinetoCTrace,
    events: Box<EventArray>,
}

impl ProfilerResult {
    /// Takes ownership of the kineto trace.
    ///
    /// # Safety
    /// `trace` must come from `eprof_kineto_finish` and not be used again.
    pub unsafe fn new(trace_start_ns: u64, trace: *mut KinetoCTrace, events: Box<EventArray>) -> Self {
        ProfilerResult { trace_start_ns, trace, events }
    }

    pub fn events(&mut self) -> &mut EventArray {
        &mut self.events
    }

    /// Writes the chrome trace. False if the path is not representable as C
    /// string, or if kineto refused it.
    pub fn save(&mut self, path: &Path) -> bool {
        let Ok(c) = CString::new(path.as_os_str().as_encoded_bytes()) else {
            return false;
        };
        // SAFETY: `trace` is ours and `c` is NUL-terminated for the call.
        unsafe { kineto_c_trace_save(self.trace, c.as_ptr()) != 0 }
    }
}

impl Drop for ProfilerResult {
    fn drop(&mut self) {
        // SAFETY: the trace is ours, freed once, and not reachable after this.
        unsafe { kineto_c_trace_free(self.trace) };
    }
}
