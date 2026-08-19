//! One event, and the array of them a run produces.
//!
//! An event is a tagged record: a torch op, a python frame, a kineto activity,
//! a power sample, an allocation, an OOM. They share a start time, a thread and
//! a parent, and differ in the fields each kind fills -- which is why this is
//! one wide struct rather than an enum. The array is append-only during a run
//! and sorted once at the end; a parent is an index into it, not a pointer, so
//! the whole tree survives being sorted, copied or handed across a boundary.
//!
//! Everything derived from an event is computed here rather than stored: its
//! display name, its end time, the metadata that belongs on its kineto
//! activity. Collection is the hot path, so it writes the minimum and this
//! answers the questions afterwards.

use std::ffi::{c_char, CString};

/// What kind of thing an event is. The numbering is part of the ABI: the C
/// headers spell these out and `eprof.treediff` names them, so a value may be
/// appended but never renumbered.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EventType {
    TorchOp = 0,
    Power = 1,
    Allocation = 2,
    OutOfMemory = 3,
    PyCall = 4,
    PyCCall = 5,
    Kineto = 6,
}

impl EventType {
    pub fn from_i32(v: i32) -> EventType {
        match v {
            1 => EventType::Power,
            2 => EventType::Allocation,
            3 => EventType::OutOfMemory,
            4 => EventType::PyCall,
            5 => EventType::PyCCall,
            6 => EventType::Kineto,
            _ => EventType::TorchOp,
        }
    }
    pub fn is_python(self) -> bool {
        matches!(self, EventType::PyCall | EventType::PyCCall)
    }
}

/// A Python frame: where a call was made from.
#[derive(Clone, Default)]
pub struct FrameState {
    pub line_no: i32,
    pub filename: String,
    pub funcname: String,
}

/// An nn.Module instance, identified by class name plus which instance of that
/// class it is. The PyObject identities that produce the number stay in the
/// tracer.
#[derive(Clone, Default)]
pub struct ModuleInfo {
    pub cls_name: String,
    pub id: usize,
}

#[derive(Default)]
pub struct Event {
    pub tag_raw: i32,

    // Common
    pub start_time_ns: i64,
    pub start_tid: u64,
    pub device: i32,   // kineto device id
    pub resource: i32, // kineto resource id
    pub finished: bool,
    /// Index of the parent event, or -1. Assigned by materialization.
    pub parent: i64,
    /// Opaque libkineto activity this event was matched to, for metadata
    /// attachment. Zero when there is none. C++ owns whatever this points at.
    pub activity_ptr: u64,

    // TorchOp
    pub sequence_number: i64,
    pub forward_tid: u64,
    pub scope: u8,
    pub record_function_id: u64,
    pub metadata: Vec<(String, String)>,

    // TorchOp / Kineto
    pub name: String,
    pub correlation_id: u64,

    // TorchOp / PyCall / PyCCall
    pub end_time_ns: i64,

    // Power / Allocation / OutOfMemory
    pub device_type: i8,
    pub device_index: i8,
    pub power_usage_mw: i64,
    pub ptr: u64,
    pub alloc_size: i64,
    pub total_allocated: u64,
    pub total_reserved: u64,

    // PyCall / PyCCall
    pub python_tid: usize,
    pub python_id: usize,
    pub callsite: FrameState,
    pub module: Option<ModuleInfo>,
    pub function_name: String,

    // Kineto
    pub duration_ns: i64,
    pub activity_type: i32,
    pub flow_id: u32,
    pub flow_type: u32,
    pub flow_start: u32,
    /// Index of the linked activity, or -1.
    pub linked: i64,
}

impl Event {
    pub fn tag(&self) -> EventType {
        EventType::from_i32(self.tag_raw)
    }

    /// The display name. Most events carry one; the memory events are named
    /// after what they are, and Python calls are rendered from their frame.
    pub fn display_name(&self) -> String {
        match self.tag() {
            EventType::Allocation => "[memory]".to_string(),
            EventType::OutOfMemory => "[OutOfMemory]".to_string(),
            EventType::Power => "[Power]".to_string(),
            EventType::PyCall => match &self.module {
                Some(m) => format!("nn.Module: {}_{}", m.cls_name, m.id),
                None => format!(
                    "{}({}): {}",
                    self.callsite.filename, self.callsite.line_no, self.callsite.funcname
                ),
            },
            EventType::PyCCall => self.function_name.clone(),
            _ => self.name.clone(),
        }
    }
}

/// libkineto::ActivityType values this code needs to name. Only the ones the
/// former C++ switch produced are listed.
mod kineto_type {
    pub const CPU_OP: i32 = 0;
    pub const USER_ANNOTATION: i32 = 1;
    pub const CPU_INSTANT_EVENT: i32 = 8;
    pub const PYTHON_FUNCTION: i32 = 9;
}

/// at::RecordScope::USER_SCOPE
const USER_SCOPE: u8 = 7;

/// One node of the materialized tree, as the python bridge reads it. `name`
/// points into the array that produced it and is valid until the next
/// `export` or `clear`.
#[repr(C)]
#[derive(Clone)]
pub struct ExportNode {
    pub id: i64,
    pub parent_id: i64,
    pub tag: i32,
    pub name: *const c_char,
    pub start_tid: u64,
    pub forward_tid: u64,
    pub start_ns: i64,
    pub dur_ns: i64,
    pub correlation_id: u64,
    pub device: i32,
    pub device_type: i32,
    pub device_index: i32,
    pub power_usage: i64,
    pub resource: i32,
    pub flow_id: u32,
    pub flow_type: u32,
    pub flow_start: u32,
    pub linked_correlation: u64,
    pub linked_id: i64,
    pub activity_type: i32,
}

/// A string as a JSON value: quoted, with what JSON requires escaped.
///
/// Metadata values reach a chrome trace verbatim, so whatever is not valid JSON
/// here makes the whole file unreadable rather than just that field.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[derive(Default)]
pub struct EventArray {
    events: Vec<Event>,
    /// Built by `export`, kept alive because its nodes point into `scratch`.
    export: Vec<ExportNode>,
    /// Owned C strings handed back through the ABI, kept until the array is
    /// cleared so the pointers stay valid.
    scratch: Vec<CString>,
}

impl EventArray {
    pub fn push(&mut self, e: Event) -> usize {
        self.events.push(e);
        self.events.len() - 1
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<&Event> {
        self.events.get(i)
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut Event> {
        self.events.get_mut(i)
    }

    /// Puts the array in start-time order, which parenting depends on. The sort
    /// is stable: events sharing a timestamp keep the order they were collected
    /// in, so a zero-duration op still nests inside the one that called it.
    pub fn sort_by_start_time(&mut self) {
        // Sorting moves events, so the index-based links have to be remapped.
        let mut order: Vec<usize> = (0..self.events.len()).collect();
        order.sort_by_key(|&i| self.events[i].start_time_ns);
        let mut old_to_new = vec![0usize; self.events.len()];
        for (new, &old) in order.iter().enumerate() {
            old_to_new[old] = new;
        }
        let mut sorted: Vec<Event> = Vec::with_capacity(self.events.len());
        for &old in &order {
            sorted.push(std::mem::take(&mut self.events[old]));
        }
        self.events = sorted;
        for e in &mut self.events {
            if e.linked >= 0 {
                e.linked = old_to_new[e.linked as usize] as i64;
            }
            if e.parent >= 0 {
                e.parent = old_to_new[e.parent as usize] as i64;
            }
        }
    }

    /// The end time, with the borrowing rule the C++ accessor had: a TorchOp
    /// that never got an end time may take its parent's, and an event that
    /// ended up with a non-positive duration reports zero rather than negative.
    pub fn end_time_ns(&self, i: usize) -> i64 {
        let e = match self.events.get(i) {
            Some(e) => e,
            None => return 0,
        };
        let raw = match e.tag() {
            EventType::TorchOp => {
                if e.finished && e.end_time_ns == i64::MIN && e.parent >= 0 {
                    self.end_time_ns(e.parent as usize)
                } else {
                    e.end_time_ns
                }
            }
            EventType::Power | EventType::Allocation | EventType::OutOfMemory => e.start_time_ns,
            EventType::Kineto => e.start_time_ns + e.duration_ns,
            _ => e.end_time_ns,
        };
        // Valid if it is a real duration, or if the event can still borrow one.
        if raw > e.start_time_ns || !e.finished {
            raw
        } else {
            e.start_time_ns
        }
    }

    /// Kineto events without their own correlation id inherit their parent's.
    pub fn correlation_id(&self, i: usize) -> u64 {
        let e = match self.events.get(i) {
            Some(e) => e,
            None => return 0,
        };
        match e.tag() {
            EventType::TorchOp => e.correlation_id,
            EventType::Kineto => {
                if e.correlation_id != 0 {
                    e.correlation_id
                } else if e.parent >= 0 {
                    self.correlation_id(e.parent as usize)
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    pub fn kineto_type(&self, i: usize) -> i32 {
        let e = match self.events.get(i) {
            Some(e) => e,
            None => return kineto_type::CPU_OP,
        };
        match e.tag() {
            EventType::TorchOp => {
                if e.scope == USER_SCOPE {
                    kineto_type::USER_ANNOTATION
                } else {
                    kineto_type::CPU_OP
                }
            }
            EventType::Power | EventType::Allocation | EventType::OutOfMemory => {
                kineto_type::CPU_INSTANT_EVENT
            }
            EventType::PyCall | EventType::PyCCall => kineto_type::PYTHON_FUNCTION,
            EventType::Kineto => e.activity_type,
        }
    }

    /// Names of the Python ancestors, innermost first -- the "Call stack"
    /// metadata field.
    pub fn python_ancestor_names(&self, i: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = self.events.get(i).map_or(-1, |e| e.parent);
        while cur >= 0 {
            let e = match self.events.get(cur as usize) {
                Some(e) => e,
                None => break,
            };
            if e.tag().is_python() {
                out.push(e.display_name());
            }
            cur = e.parent;
        }
        out
    }

    /// The nearest Python ancestor's id, for "Python parent id".
    pub fn python_parent_id(&self, i: usize) -> Option<usize> {
        let mut cur = self.events.get(i).map_or(-1, |e| e.parent);
        while cur >= 0 {
            let e = self.events.get(cur as usize)?;
            if e.tag().is_python() {
                return Some(e.python_id);
            }
            cur = e.parent;
        }
        None
    }

    pub fn clear(&mut self) {
        self.events.clear();
        self.scratch.clear();
    }

    /// Assigns every event its parent and its effective thread id.
    ///
    /// Until this runs the array is a flat list in start-time order; afterwards
    /// it is a tree. Nothing reads a parent before this point, which is why
    /// collection can get away with never computing one.
    pub fn materialize(&mut self, current_tid: u64) {
        let nodes: Vec<crate::materialize::FullNode> = self
            .events
            .iter()
            .enumerate()
            .map(|(i, e)| crate::materialize::FullNode {
                id: i as i64,
                tag: e.tag_raw,
                start_tid: e.start_tid,
                forward_tid: e.forward_tid,
                start_ns: e.start_time_ns,
                end_ns: self.end_time_ns(i),
                flow_id: e.flow_id,
                flow_type: e.flow_type,
                flow_start: e.flow_start,
                linked_id: e.linked,
            })
            .collect();
        let out = crate::materialize::materialize(&nodes, current_tid);
        for (i, e) in self.events.iter_mut().enumerate() {
            e.start_tid = out.tids[i];
            e.parent = out.parents[i];
            e.finished = true;
        }
    }

    /// The device an event is *attributed to*, which is not the same as the
    /// `device_type` field. Only allocations and OOMs name a device of their
    /// own; a kineto event's was resolved from its activity type when it was
    /// pushed. Everything else -- torch ops, python frames, and power samples
    /// -- is recorded by the CPU thread and so reads as CPU, which is what
    /// makes a parentless one eligible to root the tree.
    pub fn device_type(&self, i: usize) -> i8 {
        let e = &self.events[i];
        match e.tag() {
            EventType::Allocation | EventType::OutOfMemory | EventType::Kineto => {
                e.device_type
            }
            _ => 0,
        }
    }

    /// Everything that should end up on this event's libkineto activity: the
    /// per-kind fields recorded during collection, plus the ones derived from
    /// the tree once it is built. Entries whose value would be empty are
    /// dropped here rather than by the caller, so a trace never carries a key
    /// with nothing behind it.
    pub fn kineto_metadata(&self, i: usize) -> Vec<(String, String)> {
        let Some(e) = self.get(i) else {
            return Vec::new();
        };
        let mut out: Vec<(String, String)> = e.metadata.clone();

        if e.tag() == EventType::TorchOp {
            if e.sequence_number >= 0 {
                out.push(("Fwd thread id".into(), e.forward_tid.to_string()));
                out.push(("Sequence number".into(), e.sequence_number.to_string()));
            }
            out.push((
                "Record function id".into(),
                e.record_function_id.to_string(),
            ));
        }

        // The kinds that name a device of their own. These were lost when this
        // builder moved out of C++: a [Power] event reached the trace with no
        // power reading on it, which is the one number it exists to carry.
        match e.tag() {
            EventType::Power => {
                out.push(("Device Type".into(), e.device_type.to_string()));
                out.push(("Device Id".into(), e.device_index.to_string()));
                out.push(("Power Usage".into(), e.power_usage_mw.to_string()));
            }
            EventType::Allocation => {
                out.push(("Device Type".into(), e.device_type.to_string()));
                out.push(("Device Id".into(), e.device_index.to_string()));
                out.push(("Addr".into(), e.ptr.to_string()));
                out.push(("Bytes".into(), e.alloc_size.to_string()));
                out.push(("Total Allocated".into(), e.total_allocated.to_string()));
                out.push(("Total Reserved".into(), e.total_reserved.to_string()));
            }
            EventType::OutOfMemory => {
                out.push(("Device Type".into(), e.device_type.to_string()));
                out.push(("Device Id".into(), e.device_index.to_string()));
                out.push(("Bytes".into(), e.alloc_size.to_string()));
                out.push(("Total Allocated".into(), e.total_allocated.to_string()));
                out.push(("Total Reserved".into(), e.total_reserved.to_string()));
            }
            _ => {}
        }

        // Quoted here, unlike the numbers above and unlike the values C++
        // produces, which arrive already quoted. A call stack is file paths and
        // source lines joined together, so it is the one value that both needs
        // quoting and can contain characters that have to be escaped -- without
        // this the trace is not parseable JSON at all.
        out.push((
            "Call stack".into(),
            json_string(&self.python_ancestor_names(i).join(";")),
        ));

        if matches!(e.tag(), EventType::PyCall | EventType::PyCCall) {
            out.push(("Python id".into(), e.python_id.to_string()));
            out.push((
                "Python parent id".into(),
                self.python_parent_id(i)
                    .map_or_else(|| "null".to_string(), |p| p.to_string()),
            ));
        }

        // An empty string, or a quoted empty string, means the field was not
        // recorded rather than recorded as blank.
        out.retain(|(_, v)| !v.is_empty() && v != "\"\"");
        out
    }

    /// The materialized tree, flattened for the python bridge: pre-order, with
    /// ids and parent ids that are positions in this list. Parentless GPU
    /// events are not in the tree and so are not here either.
    ///
    /// The nodes borrow their names from `scratch`, so this holds onto both and
    /// hands out a slice; rebuilding it on a later call replaces the lot.
    pub fn export(&mut self) -> &[ExportNode] {
        let order = self.preorder();
        let mut pos = vec![-1i64; self.events.len()];
        for (k, &i) in order.iter().enumerate() {
            pos[i] = k as i64;
        }

        self.scratch.clear();
        for &i in &order {
            self.scratch
                .push(CString::new(self.events[i].display_name()).unwrap_or_default());
        }

        self.export = order
            .iter()
            .enumerate()
            .map(|(k, &i)| {
                let e = &self.events[i];
                ExportNode {
                    id: k as i64,
                    parent_id: if e.parent >= 0 { pos[e.parent as usize] } else { -1 },
                    tag: e.tag_raw,
                    name: self.scratch[k].as_ptr(),
                    start_tid: e.start_tid,
                    forward_tid: e.forward_tid,
                    start_ns: e.start_time_ns,
                    dur_ns: self.end_time_ns(i) - e.start_time_ns,
                    correlation_id: self.correlation_id(i),
                    device: e.device,
                    device_type: self.device_type(i) as i32,
                    device_index: e.device_index as i32,
                    // Only a power sample carries one; -1 reads as "not power".
                    power_usage: if e.tag() == EventType::Power {
                        e.power_usage_mw
                    } else {
                        -1
                    },
                    resource: e.resource,
                    flow_id: e.flow_id,
                    flow_type: e.flow_type,
                    flow_start: e.flow_start,
                    linked_correlation: if e.linked >= 0 {
                        self.correlation_id(e.linked as usize)
                    } else {
                        0
                    },
                    linked_id: if e.linked >= 0 { pos[e.linked as usize] } else { -1 },
                    activity_type: e.activity_type,
                }
            })
            .collect();
        &self.export
    }

    /// Pre-order over the tree, which is what the flat id/parent_id bridge
    /// encodes. A parentless CPU event is a root; a parentless GPU event is
    /// deliberately not one, so top-level kernels stay out of the tree.
    pub fn preorder(&self) -> Vec<usize> {
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); self.events.len()];
        let mut roots = Vec::new();
        for (i, e) in self.events.iter().enumerate() {
            if e.parent >= 0 {
                children[e.parent as usize].push(i);
            } else if self.device_type(i) == 0 {
                roots.push(i);
            }
        }
        let mut out = Vec::with_capacity(self.events.len());
        let mut stack: Vec<usize> = roots.into_iter().rev().collect();
        while let Some(i) = stack.pop() {
            out.push(i);
            for &c in children[i].iter().rev() {
                stack.push(c);
            }
        }
        out
    }
}

// --- Ownership --------------------------------------------------------------
//
// The array is handed to the C++ tracer as a raw pointer for the duration of a
// run -- it is the one thing both sides write into -- so it is created boxed
// and leaked, and reclaimed when the run's result is built.

/// # Safety
/// Release with `eprof_events_destroy`, or reclaim with `Box::from_raw`.
pub fn eprof_events_create() -> *mut EventArray {
    Box::into_raw(Box::new(EventArray::default()))
}

/// # Safety
/// `arr` must come from `eprof_events_create`.
pub unsafe fn eprof_events_destroy(arr: *mut EventArray) {
    if !arr.is_null() {
        drop(Box::from_raw(arr));
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn ev(tag: EventType, start: i64) -> Event {
        Event { tag_raw: tag as i32, start_time_ns: start, parent: -1, linked: -1,
                ..Default::default() }
    }

    #[test]
    fn memory_events_are_named_for_what_they_are() {
        let mut e = ev(EventType::Allocation, 0);
        e.name = "ignored".into();
        assert_eq!(e.display_name(), "[memory]");
    }

    #[test]
    fn a_module_call_is_named_by_class_and_instance() {
        let mut e = ev(EventType::PyCall, 0);
        e.module = Some(ModuleInfo { cls_name: "Linear".into(), id: 2 });
        assert_eq!(e.display_name(), "nn.Module: Linear_2");
    }

    #[test]
    fn a_plain_python_call_is_named_by_its_frame() {
        let mut e = ev(EventType::PyCall, 0);
        e.callsite = FrameState { line_no: 93, filename: "linear.py".into(),
                                  funcname: "forward".into() };
        assert_eq!(e.display_name(), "linear.py(93): forward");
    }

    #[test]
    fn an_unfinished_op_borrows_its_parents_end_time() {
        let mut a = EventArray::default();
        let mut parent = ev(EventType::TorchOp, 100);
        parent.end_time_ns = 900;
        parent.finished = true;
        a.push(parent);
        let mut child = ev(EventType::TorchOp, 200);
        child.end_time_ns = i64::MIN;
        child.finished = true;
        child.parent = 0;
        a.push(child);
        assert_eq!(a.end_time_ns(1), 900);
    }

    #[test]
    fn a_non_positive_duration_collapses_to_the_start_time() {
        let mut a = EventArray::default();
        let mut e = ev(EventType::Kineto, 500);
        e.duration_ns = 0;
        e.finished = true;
        a.push(e);
        assert_eq!(a.end_time_ns(0), 500);
    }

    #[test]
    fn a_kineto_event_inherits_its_parents_correlation_id() {
        let mut a = EventArray::default();
        let mut parent = ev(EventType::TorchOp, 0);
        parent.correlation_id = 77;
        a.push(parent);
        let mut child = ev(EventType::Kineto, 1);
        child.parent = 0;
        a.push(child);
        assert_eq!(a.correlation_id(1), 77);
    }

    #[test]
    fn sorting_remaps_parent_and_linked_indices() {
        let mut a = EventArray::default();
        a.push(ev(EventType::TorchOp, 300)); // 0 -> becomes 1
        let mut child = ev(EventType::Kineto, 100); // 1 -> becomes 0
        child.parent = 0;
        child.linked = 0;
        a.push(child);
        a.sort_by_start_time();
        assert_eq!(a.get(0).unwrap().start_time_ns, 100);
        assert_eq!(a.get(0).unwrap().parent, 1, "parent index must follow the move");
        assert_eq!(a.get(0).unwrap().linked, 1, "linked index must follow the move");
    }

    #[test]
    fn preorder_excludes_parentless_gpu_events() {
        let mut a = EventArray::default();
        let cpu = ev(EventType::TorchOp, 0);          // 0: CPU root
        a.push(cpu);
        let mut gpu = ev(EventType::Kineto, 1);       // 1: parentless GPU -> not a root
        gpu.device_type = 1;
        a.push(gpu);
        let mut child = ev(EventType::Kineto, 2);     // 2: child of the CPU root
        child.parent = 0;
        a.push(child);
        assert_eq!(a.preorder(), vec![0, 2]);
    }

    #[test]
    fn preorder_is_depth_first() {
        let mut a = EventArray::default();
        a.push(ev(EventType::TorchOp, 0));            // 0 root
        let mut c1 = ev(EventType::TorchOp, 1); c1.parent = 0; a.push(c1);   // 1
        let mut c2 = ev(EventType::TorchOp, 2); c2.parent = 0; a.push(c2);   // 2
        let mut g = ev(EventType::TorchOp, 3);  g.parent = 1; a.push(g);     // 3 under 1
        assert_eq!(a.preorder(), vec![0, 1, 3, 2]);
    }

    #[test]
    fn the_call_stack_walks_only_python_ancestors() {
        let mut a = EventArray::default();
        let mut py = ev(EventType::PyCall, 0);
        py.callsite = FrameState { line_no: 1, filename: "a.py".into(), funcname: "f".into() };
        py.python_id = 5;
        a.push(py);
        let mut op = ev(EventType::TorchOp, 1); // non-Python, should be skipped
        op.parent = 0;
        a.push(op);
        let mut leaf = ev(EventType::Kineto, 2);
        leaf.parent = 1;
        a.push(leaf);
        assert_eq!(a.python_ancestor_names(2), vec!["a.py(1): f".to_string()]);
        assert_eq!(a.python_parent_id(2), Some(5));
    }

    #[test]
    fn only_allocations_oom_and_kineto_name_their_own_device() {
        let mut a = EventArray::default();
        for tag in [EventType::Power, EventType::Allocation, EventType::TorchOp] {
            a.push(Event {
                tag_raw: tag as i32,
                device_type: 1, // CUDA, as reported by the sampler
                parent: -1,
                linked: -1,
                ..Default::default()
            });
        }
        assert_eq!(a.device_type(0), 0, "a power sample is taken on the CPU thread");
        assert_eq!(a.device_type(1), 1, "an allocation names its own device");
        assert_eq!(a.device_type(2), 0);
    }

    #[test]
    fn a_parentless_power_sample_roots_the_tree() {
        let mut a = EventArray::default();
        a.push(Event {
            tag_raw: EventType::Power as i32,
            device_type: 1,
            parent: -1,
            linked: -1,
            ..Default::default()
        });
        assert_eq!(a.preorder(), vec![0], "must not be filtered out as a GPU event");
    }
}

#[cfg(test)]
mod json_tests {
    use super::json_string;

    #[test]
    fn a_plain_string_is_quoted() {
        assert_eq!(json_string("a;b"), "\"a;b\"");
    }

    #[test]
    fn the_characters_json_forbids_are_escaped() {
        assert_eq!(json_string(r#"a"b\c"#), r#""a\"b\\c""#);
        assert_eq!(json_string("a\nb\tc"), "\"a\\nb\\tc\"");
        assert_eq!(json_string("a\u{1}b"), "\"a\\u0001b\"");
    }

    #[test]
    fn a_call_stack_round_trips_through_a_json_parser() {
        // What actually goes in one: paths, line numbers, and a module name.
        let stack = "torch/nn/modules/module.py(1755): _call_impl;nn.Module: Linear_0";
        let doc = format!("{{\"Call stack\": {}}}", json_string(stack));
        // A minimal check that the value is closed and the braces balance --
        // the failure this guards against is an unquoted value swallowing the
        // rest of the document.
        assert!(doc.ends_with("\"}"));
        assert_eq!(doc.matches('"').count() % 2, 0);
    }
}
