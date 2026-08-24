
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::sync::Mutex;

use eprof_storage::event::{Event, EventArray, EventType};
use crate::inputs::OpInputs;
use crate::ops::{OpStore, UNSET_TIME};
use crate::pods::{PodEventStore, TimeConverter};

/// The ids libkineto places a thread's activities by.
#[derive(Clone, Copy, Default)]
pub struct KinetoIds {
    pub device: i32,
    pub resource: i32,
}

/// One recording thread's captured work.
pub struct Subqueue {
    pub tid: u64,
    pub kineto: KinetoIds,
    pub ops: OpStore,
    pub pods: PodEventStore,
    pub inputs: OpInputs,
    pub extra_meta: Vec<Vec<(String, String)>>,
    pub kwinputs: Vec<Vec<(String, String)>>,
    /// (trace key, raw timestamp) for each python frame entered on this thread.
    pub py_calls: Vec<(u64, i64)>,
}

impl Subqueue {
    fn new(tid: u64, kineto: KinetoIds) -> Self {
        Subqueue {
            tid,
            kineto,
            ops: OpStore::default(),
            pods: PodEventStore::default(),
            inputs: OpInputs::default(),
            extra_meta: Vec::new(),
            kwinputs: Vec::new(),
            py_calls: Vec::new(),
        }
    }

    /// Turns this thread's captured ops into events on `arr`.
    fn materialize(&mut self, arr: &mut EventArray, convert: &dyn Fn(i64) -> i64) {
        self.ops.plumb_autograd_sequence_numbers();
        self.ops.normalize_accumulate_grad_names();

        for i in 0..self.ops.len() {
            let Some(op) = self.ops.take(i) else {
                continue;
            };
            let e = arr.push(Event {
                tag_raw: EventType::TorchOp as i32,
                start_time_ns: convert(op.start_time),
                start_tid: self.tid,
                device: self.kineto.device,
                resource: self.kineto.resource,
                end_time_ns: convert(op.end_time),
                sequence_number: op.sequence_number,
                forward_tid: op.forward_tid,
                scope: op.scope,
                record_function_id: op.record_function_id,
                correlation_id: op.correlation_id,
                name: op.name,
                parent: -1,
                linked: -1,
                ..Default::default()
            });

            // The input fields come off the stream that recorded them, in the
            // same order the ops were captured.
            let inputs = self.inputs.next_metadata();
            if let Some(ev) = arr.get_mut(e) {
                ev.metadata.extend(inputs);
                if let Some(kw) = self.kwinputs.get_mut(i) {
                    ev.metadata.append(kw);
                }
                if let Some(extra) = self.extra_meta.get_mut(i) {
                    ev.metadata.append(extra);
                }
            }
        }

        self.ops.clear();
        self.inputs.clear();
        self.extra_meta.clear();
        self.kwinputs.clear();
    }
}

/// Everything an active profiler collects into.
pub struct RecordQueue {
    pub id: u32,
    subqueues: Mutex<HashMap<u64, Box<Subqueue>>>,
    /// The array this run materializes into.
    pub events: *mut EventArray,
    pub python_enters: Vec<eprof_storage::frames::EnterRecord>,
}

// The queue is shared across recording threads; the subqueue map is behind a
// mutex and the subqueues themselves are only touched by their owning thread.
unsafe impl Send for RecordQueue {}
unsafe impl Sync for RecordQueue {}

impl RecordQueue {
    /// The subqueue for `tid`, creating it on first use.
    fn subqueue(&self, tid: u64, kineto: KinetoIds) -> *mut Subqueue {
        let mut map = self.subqueues.lock().expect("subqueue map poisoned");
        let entry = map
            .entry(tid)
            .or_insert_with(|| Box::new(Subqueue::new(tid, kineto)));
        &mut **entry as *mut Subqueue
    }
}

// --- Rust API ----------------------------------------------------------------
// What the run driver uses. The C ABI below is only what C++ calls.

pub unsafe fn create(events: *mut EventArray) -> *mut RecordQueue {
    eprof_queue_create(events)
}

pub unsafe fn destroy(queue: *mut RecordQueue) {
    eprof_queue_destroy(queue);
}

pub unsafe fn merge(queue: *mut RecordQueue, convert: TimeConverter, ctx: *mut c_void) {
    eprof_queue_merge(queue, convert, ctx);
}

extern "C" {
    fn eprof_current_thread_id() -> u64;
}

/// The calling thread's subqueue, created on first use.
pub unsafe fn subqueue_for_current_thread(queue: *mut RecordQueue) -> *mut Subqueue {
    thread_local! {
        static CACHE: std::cell::Cell<(u32, *mut Subqueue)> =
            const { std::cell::Cell::new((0, std::ptr::null_mut())) };
    }
    let id = (*queue).id;
    let (cached_id, cached) = CACHE.with(|c| c.get());
    if id != 0 && id == cached_id {
        return cached;
    }
    let (device, resource) = eprof_kineto::ids();
    eprof_kineto::record_thread_info();
    let sq = (*queue).subqueue(
        eprof_current_thread_id(),
        KinetoIds { device, resource },
    );
    CACHE.with(|c| c.set((id, sq)));
    sq
}

// --- C ABI ------------------------------------------------------------------

unsafe fn read(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

static NEXT_QUEUE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub unsafe fn eprof_queue_create(events: *mut EventArray) -> *mut RecordQueue {
    Box::into_raw(Box::new(RecordQueue {
        id: NEXT_QUEUE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1,
        subqueues: Mutex::new(HashMap::new()),
        events,
        python_enters: Vec::new(),
    }))
}

pub unsafe fn eprof_queue_destroy(queue: *mut RecordQueue) {
    if !queue.is_null() {
        drop(Box::from_raw(queue));
    }
}

pub unsafe fn eprof_queue_id(queue: *const RecordQueue) -> u32 {
    if queue.is_null() {
        0
    } else {
        (*queue).id
    }
}

/// The subqueue for `tid`, created on first use.
pub unsafe fn eprof_queue_subqueue(
    queue: *const RecordQueue,
    tid: u64,
    device: i32,
    resource: i32,
) -> *mut Subqueue {
    if queue.is_null() {
        return std::ptr::null_mut();
    }
    (*queue).subqueue(tid, KinetoIds { device, resource })
}

/// Adds to the metadata of the op most recently begun.
#[no_mangle]
pub unsafe extern "C" fn eprof_subqueue_add_op_metadata(
    sq: *mut Subqueue,
    key: *const c_char,
    value: *const c_char,
    is_kwinput: i32,
) {
    if sq.is_null() {
        return;
    }
    let list = if is_kwinput != 0 {
        (*sq).kwinputs.last_mut()
    } else {
        (*sq).extra_meta.last_mut()
    };
    if let Some(list) = list {
        list.push((read(key), read(value)));
    }
}

pub unsafe fn eprof_queue_merge(
    queue: *mut RecordQueue,
    convert: TimeConverter,
    ctx: *mut c_void,
) {
    if queue.is_null() {
        return;
    }
    let queue = &mut *queue;
    if queue.events.is_null() {
        return;
    }
    let arr = &mut *queue.events;

    // An op whose exit callback never ran has no end time of its own; leaving
    // the sentinel lets materialization borrow one from its parent.
    let convert_fn = |t: i64| {
        if t == UNSET_TIME {
            i64::MIN
        } else {
            convert(ctx, t)
        }
    };

    let mut map = queue.subqueues.lock().expect("subqueue map poisoned");
    for sq in map.values_mut() {
        sq.materialize(arr, &convert_fn);
        crate::pods::drain_into(
            &mut sq.pods,
            arr,
            sq.tid,
            sq.kineto.device,
            sq.kineto.resource,
            &convert_fn,
        );
        for &(key, t) in &sq.py_calls {
            queue.python_enters.push(eprof_storage::frames::EnterRecord {
                key,
                system_tid: sq.tid,
                device: sq.kineto.device,
                resource: sq.kineto.resource,
                start_ns: convert_fn(t),
            });
        }
        sq.py_calls.clear();
    }
}

/// The gathered python frame entries, for the tracer's replay.
#[no_mangle]
pub unsafe extern "C" fn eprof_queue_python_enters(
    queue: *mut RecordQueue,
    out_len: *mut usize,
) -> *mut eprof_storage::frames::EnterRecord {
    if queue.is_null() {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return std::ptr::null_mut();
    }
    let enters = &mut (*queue).python_enters;
    if !out_len.is_null() {
        *out_len = enters.len();
    }
    enters.as_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn eprof_queue_push_python_enter(
    queue: *mut RecordQueue,
    key: u64,
    system_tid: u64,
    start_ns: i64,
) {
    if !queue.is_null() {
        (*queue).python_enters.push(eprof_storage::frames::EnterRecord {
            key,
            system_tid,
            device: 0,
            resource: 0,
            start_ns,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert_identity(t: i64) -> i64 {
        t
    }

    #[test]
    fn a_thread_gets_the_same_subqueue_every_time() {
        let mut arr = EventArray::default();
        let q = unsafe { eprof_queue_create(&mut arr) };
        let ids = (7, 0, 0);
        let a = unsafe { eprof_queue_subqueue(q, ids.0, ids.1, ids.2) };
        let b = unsafe { eprof_queue_subqueue(q, ids.0, ids.1, ids.2) };
        assert_eq!(a, b);
        let c = unsafe { eprof_queue_subqueue(q, 8, 0, 0) };
        assert_ne!(a, c, "a different thread gets its own");
        unsafe { eprof_queue_destroy(q) };
    }

    #[test]
    fn a_subqueue_pointer_survives_other_threads_arriving() {
        // The caller caches it per thread, so growth of the map must not move
        // one already handed out.
        let mut arr = EventArray::default();
        let q = unsafe { eprof_queue_create(&mut arr) };
        let first = unsafe { eprof_queue_subqueue(q, 0, 0, 0) };
        for tid in 1..64 {
            unsafe { eprof_queue_subqueue(q, tid, 0, 0) };
        }
        assert_eq!(first, unsafe { eprof_queue_subqueue(q, 0, 0, 0) });
        unsafe { eprof_queue_destroy(q) };
    }

    #[test]
    fn queue_ids_are_never_reused() {
        let mut arr = EventArray::default();
        let a = unsafe { eprof_queue_create(&mut arr) };
        let b = unsafe { eprof_queue_create(&mut arr) };
        assert_ne!(unsafe { eprof_queue_id(a) }, unsafe { eprof_queue_id(b) });
        unsafe {
            eprof_queue_destroy(a);
            eprof_queue_destroy(b);
        }
    }

    #[test]
    fn materializing_an_op_carries_its_fields_and_metadata_across() {
        let mut sq = Subqueue::new(3, KinetoIds { device: 1, resource: 2 });
        let (i, _) = sq.ops.push("aten::mm", 5, 9, 0, 11, 100);
        sq.ops.set_end(i, 200, 3);
        sq.inputs.push_tensor("float".into(), true, &[2], &[1]);
        sq.inputs.push_tag(crate::inputs::TAG_TERMINATOR);
        sq.extra_meta.push(vec![("Collective name".into(), "\"all_reduce\"".into())]);
        sq.kwinputs.push(vec![("stream".into(), "7".into())]);

        let mut arr = EventArray::default();
        sq.materialize(&mut arr, &convert_identity);

        assert_eq!(arr.len(), 1);
        let e = arr.get(0).unwrap();
        assert_eq!(e.name, "aten::mm");
        assert_eq!(e.start_time_ns, 100);
        assert_eq!(e.end_time_ns, 200);
        assert_eq!(e.sequence_number, 5);
        assert_eq!(e.start_tid, 3);
        assert_eq!((e.device, e.resource), (1, 2));
        let keys: Vec<&str> = e.metadata.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"Input Dims"));
        assert!(keys.contains(&"stream"));
        assert!(keys.contains(&"Collective name"));
    }

    #[test]
    fn metadata_is_matched_to_ops_positionally() {
        // An op with no metadata of its own still needs its slot, or every
        // later op takes the previous one's.
        let mut sq = Subqueue::new(0, KinetoIds::default());
        for (n, name) in ["a", "b", "c"].iter().enumerate() {
            sq.ops.push(name, -1, 0, 0, 0, n as i64);
            sq.inputs.push_tag(crate::inputs::TAG_TERMINATOR);
            sq.extra_meta.push(Vec::new());
            sq.kwinputs.push(if n == 1 {
                vec![("k".into(), "v".into())]
            } else {
                Vec::new()
            });
        }
        let mut arr = EventArray::default();
        sq.materialize(&mut arr, &convert_identity);
        assert!(arr.get(0).unwrap().metadata.is_empty());
        assert_eq!(arr.get(1).unwrap().metadata, vec![("k".into(), "v".into())]);
        assert!(arr.get(2).unwrap().metadata.is_empty());
    }

    #[test]
    fn materializing_empties_the_stores() {
        let mut sq = Subqueue::new(0, KinetoIds::default());
        sq.ops.push("op", -1, 0, 0, 0, 0);
        sq.inputs.push_tag(crate::inputs::TAG_TERMINATOR);
        sq.extra_meta.push(Vec::new());
        sq.kwinputs.push(Vec::new());
        let mut arr = EventArray::default();
        sq.materialize(&mut arr, &convert_identity);
        assert_eq!(sq.ops.len(), 0);
        sq.materialize(&mut arr, &convert_identity);
        assert_eq!(arr.len(), 1, "a second merge must not duplicate events");
    }
}
