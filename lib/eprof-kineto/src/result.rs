//! What a finished run hands back.

use std::ffi::CString;
use std::path::Path;

use eprof_storage::event::EventArray;

use super::{kineto_c_trace_free, kineto_c_trace_save, KinetoCTrace};

pub struct ProfilerResult {
    pub trace_start_ns: u64,
    trace: *mut KinetoCTrace,
    events: Box<EventArray>,
}

impl ProfilerResult {
    /// Takes ownership of the kineto trace.
    pub unsafe fn new(trace_start_ns: u64, trace: *mut KinetoCTrace, events: Box<EventArray>) -> Self {
        ProfilerResult { trace_start_ns, trace, events }
    }

    pub fn events(&mut self) -> &mut EventArray {
        &mut self.events
    }

    /// Writes the chrome trace.
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
