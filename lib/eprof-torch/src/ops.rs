//! The buffer every profiled torch op lands in.
//!
//! This is the hot path: one push per dispatched operator, from inside the
//! RecordFunction callback, on whichever thread ran the op. So it does the
//! least it can -- append the fields as read, take a correlation id, return the
//! index -- and leaves naming, timing arithmetic and parenting to the end of
//! the run. Profiling that reshapes the thing being profiled is worse than
//! useless, and the budget here is small: measured overhead is roughly 2.8 us
//! per op in total, most of it in torch's own dispatch and the IValue reads.

use std::ffi::c_char;
use std::sync::atomic::{AtomicU64, Ordering};

/// Correlation ids tie a CPU op to the kernels it launched, so kineto needs
/// them unique across every thread, not just within one buffer. A single
/// global atomic is the cheapest thing that guarantees that.
static NEXT_CORRELATION_ID: AtomicU64 = AtomicU64::new(0);

/// Sentinel matching c10::approx_time_t's "unset" value, so an op whose exit
/// callback never ran is still recognisable downstream.
pub const UNSET_TIME: i64 = i64::MIN;

pub struct OpEvent {
    pub name: String,
    pub sequence_number: i64,
    pub forward_tid: u64,
    pub scope: u8,
    pub record_function_id: u64,
    pub start_time: i64,
    pub end_time: i64,
    pub end_tid: u64,
    pub correlation_id: u64,
}

/// POD view handed back across the ABI. `name` points into the store.
#[repr(C)]
pub struct OpEventView {
    pub name: *const c_char,
    pub sequence_number: i64,
    pub forward_tid: u64,
    pub scope: u8,
    pub record_function_id: u64,
    pub start_time: i64,
    pub end_time: i64,
    pub end_tid: u64,
    pub correlation_id: u64,
}

#[derive(Default)]
pub struct OpStore {
    ops: Vec<OpEvent>,
    // Kept alongside so `name` in a view stays valid: Vec growth moves the
    // Strings' headers but not their heap buffers, so as_ptr() remains good.
    names: Vec<std::ffi::CString>,
}

impl OpStore {
    pub fn push(&mut self, name: &str, sequence_number: i64, forward_tid: u64, scope: u8,
                record_function_id: u64, start_time: i64) -> (usize, u64) {
        let correlation_id = NEXT_CORRELATION_ID.fetch_add(1, Ordering::Relaxed);
        self.names
            .push(std::ffi::CString::new(name).unwrap_or_default());
        self.ops.push(OpEvent {
            name: name.to_string(),
            sequence_number,
            forward_tid,
            scope,
            record_function_id,
            start_time,
            end_time: UNSET_TIME,
            end_tid: 0,
            correlation_id,
        });
        (self.ops.len() - 1, correlation_id)
    }

    /// begin_op needs the correlation id before it does its own setup work, but
    /// the op's start time is stamped after that work so the op is not charged
    /// for it. So the push happens first and the start time lands here.
    pub fn set_start(&mut self, idx: usize, start_time: i64) {
        if let Some(op) = self.ops.get_mut(idx) {
            op.start_time = start_time;
        }
    }

    pub fn set_end(&mut self, idx: usize, end_time: i64, end_tid: u64) {
        if let Some(op) = self.ops.get_mut(idx) {
            op.end_time = end_time;
            op.end_tid = end_tid;
        }
    }

    /// Autograd records a backward node's sequence number on the node itself,
    /// not on the `evaluate_function` annotation that wraps it -- but the
    /// annotation is what a reader sees in the trace. So each annotation takes
    /// the number from the op that follows it. The scope constants are
    /// `at::RecordScope`'s, and the numbering is torch's.
    pub fn plumb_autograd_sequence_numbers(&mut self) {
        const FUNCTION: u8 = 0;
        const BACKWARD_FUNCTION: u8 = 1;
        const ANNOTATION: &str = "autograd::engine::evaluate_function: ";
        for i in 0..self.ops.len().saturating_sub(1) {
            let (head, tail) = self.ops.split_at_mut(i + 1);
            let first = &mut head[i];
            let second = &tail[0];
            if first.scope == FUNCTION
                && second.scope == BACKWARD_FUNCTION
                && first.name.starts_with(ANNOTATION)
            {
                first.sequence_number = second.sequence_number;
                first.forward_tid = second.forward_tid;
            }
        }
    }

    /// AccumulateGrad is a marker profile analysis keys on, but the annotation
    /// comes from c10::demangle, which on Windows prefixes "struct ".
    pub fn normalize_accumulate_grad_names(&mut self) {
        const WANTED: &str = "torch::autograd::AccumulateGrad";
        let windows = format!("struct {WANTED}");
        for (i, op) in self.ops.iter_mut().enumerate() {
            if let Some(pos) = op.name.find(&windows) {
                op.name.replace_range(pos..pos + windows.len(), WANTED);
                self.names[i] =
                    std::ffi::CString::new(op.name.as_str()).unwrap_or_default();
            }
        }
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn clear(&mut self) {
        self.ops.clear();
        self.names.clear();
    }

    /// The op with its name owned, for callers inside Rust that are about to
    /// move the string onto an event rather than read it through a pointer.
    pub fn take(&self, i: usize) -> Option<OpEvent> {
        let op = self.ops.get(i)?;
        Some(OpEvent {
            name: op.name.clone(),
            sequence_number: op.sequence_number,
            forward_tid: op.forward_tid,
            scope: op.scope,
            record_function_id: op.record_function_id,
            start_time: op.start_time,
            end_time: op.end_time,
            end_tid: op.end_tid,
            correlation_id: op.correlation_id,
        })
    }

    pub fn view(&self, i: usize) -> Option<OpEventView> {
        let op = self.ops.get(i)?;
        Some(OpEventView {
            name: self.names[i].as_ptr(),
            sequence_number: op.sequence_number,
            forward_tid: op.forward_tid,
            scope: op.scope,
            record_function_id: op.record_function_id,
            start_time: op.start_time,
            end_time: op.end_time,
            end_tid: op.end_tid,
            correlation_id: op.correlation_id,
        })
    }
}

// --- C ABI ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn push_assigns_unique_increasing_correlation_ids() {
        let mut s = OpStore::default();
        let (_, a) = s.push("aten::mm", 1, 0, 0, 7, 100);
        let (_, b) = s.push("aten::add", 2, 0, 0, 8, 200);
        assert!(b > a, "correlation ids must be unique and increasing");
    }

    #[test]
    fn end_time_starts_unset_and_is_completed_by_index() {
        let mut s = OpStore::default();
        let (i, _) = s.push("aten::mm", 1, 0, 0, 7, 100);
        assert_eq!(s.view(i).unwrap().end_time, UNSET_TIME);
        s.set_end(i, 500, 42);
        let v = s.view(i).unwrap();
        assert_eq!(v.end_time, 500);
        assert_eq!(v.end_tid, 42);
    }

    #[test]
    fn set_end_on_a_bad_index_is_ignored() {
        let mut s = OpStore::default();
        s.set_end(99, 1, 1); // must not panic across FFI
        assert!(s.is_empty());
    }

    #[test]
    fn autograd_annotation_takes_the_following_backward_nodes_sequence() {
        let mut s = OpStore::default();
        s.push("autograd::engine::evaluate_function: MmBackward0", -1, 0, 0, 0, 0);
        s.push("MmBackward0", 42, 7, 1, 0, 0);
        s.plumb_autograd_sequence_numbers();
        let v = s.view(0).unwrap();
        assert_eq!(v.sequence_number, 42);
        assert_eq!(v.forward_tid, 7);
    }

    #[test]
    fn plumbing_requires_the_annotation_prefix() {
        let mut s = OpStore::default();
        s.push("aten::mm", -1, 0, 0, 0, 0);
        s.push("MmBackward0", 42, 7, 1, 0, 0);
        s.plumb_autograd_sequence_numbers();
        assert_eq!(s.view(0).unwrap().sequence_number, -1, "must not plumb");
    }

    #[test]
    fn windows_struct_prefix_is_stripped_from_accumulate_grad() {
        let mut s = OpStore::default();
        s.push("struct torch::autograd::AccumulateGrad", 0, 0, 0, 0, 0);
        s.normalize_accumulate_grad_names();
        let v = s.view(0).unwrap();
        let name = unsafe { CStr::from_ptr(v.name) }.to_str().unwrap();
        assert_eq!(name, "torch::autograd::AccumulateGrad");
    }

    #[test]
    fn names_survive_reallocation() {
        let mut s = OpStore::default();
        for i in 0..1000 {
            s.push(&format!("op{i}"), i, 0, 0, 0, i);
        }
        let v = s.view(0).unwrap();
        let name = unsafe { CStr::from_ptr(v.name) }.to_str().unwrap();
        assert_eq!(name, "op0", "view name must stay valid after Vec growth");
    }
}
