//! Pairing the frames the tracer recorded.
// key of the callsite, an exit carries only a timestamp. Pairing them is a

use std::collections::HashMap;
use std::ffi::{c_char, CStr};

use eprof_storage::event::{self as event, Event, EventArray, EventType};
use eprof_storage::frames::{EnterRecord, NO_TID};
use super::{PyCache, K_PY_CALL, K_PY_C_CALL, PY_C_CALL};

unsafe fn owned(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}


/// One recorded exit. Which frame it closes is worked out by the replay.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExitRecord {
    pub t: i64,
    pub python_tid: u64,
    /// Exits from C calls are kept apart from python ones; see above.
    pub is_c_call: i32,
}


struct Emitted {
    site: usize,
    start_ns: i64,
    start_tid: u64,
    device: i32,
    resource: i32,
    end_ns: i64,
    python_tid: u64,
}

fn replay_one_kind(
    cache: &mut PyCache,
    enters: &[EnterRecord],
    exits: &[ExitRecord],
    end_time_ns: i64,
    event_type: i32,
    out: &mut Vec<Emitted>,
) {
    let initial_len = out.len();

    // Exits in time order, so the next one to close is always the front.
    let mut pending: Vec<i64> = Vec::new();
    let mut pending_tid: Vec<u64> = Vec::new();
    {
        let want_c = event_type == K_PY_C_CALL;
        let mut sorted: Vec<&ExitRecord> = exits
            .iter()
            .filter(|e| (e.is_c_call != 0) == want_c)
            .collect();
        sorted.sort_by_key(|e| e.t);
        for e in sorted {
            pending.push(e.t);
            pending_tid.push(e.python_tid);
        }
    }
    let mut next_exit = 0usize;

    // One stack per python thread, holding positions in `out` -- `out` grows as
    // we replay, so indices are what stays valid.
    let mut stacks: HashMap<u64, Vec<usize>> = HashMap::new();

    for enter in enters {
        let Some(idx) = enter.key.checked_sub(1).map(|k| k as usize) else {
            continue;
        };
        let Some(site) = cache.sites.get(idx) else {
            continue;
        };
        let (call_type, python_tid) = (site.call_type, site.python_tid);
        if kind_of(call_type) != event_type {
            continue;
        }

        while next_exit < pending.len() && pending[next_exit] < enter.start_ns {
            if let Some(stack) = stacks.get_mut(&pending_tid[next_exit]) {
                if let Some(pos) = stack.pop() {
                    out[pos].end_ns = pending[next_exit];
                }
            }
            next_exit += 1;
        }

        out.push(Emitted {
            site: idx,
            start_ns: enter.start_ns,
            start_tid: enter.system_tid,
            device: enter.device,
            resource: enter.resource,
            end_ns: end_time_ns,
            python_tid,
        });
        stacks.entry(python_tid).or_default().push(out.len() - 1);
    }

    // Frames still open when profiling stopped run to the end of the trace.
    for stack in stacks.values() {
        for &pos in stack {
            out[pos].end_ns = end_time_ns;
        }
    }

    let mut tid_map: HashMap<u64, (u64, i32, i32)> = HashMap::new();
    for e in out[initial_len..].iter_mut().rev() {
        if e.start_tid == NO_TID && event_type == K_PY_CALL {
            let (tid, device, resource) =
                *tid_map.entry(e.python_tid).or_insert((NO_TID, 0, 0));
            e.start_tid = tid;
            e.device = device;
            e.resource = resource;
        }
        tid_map.insert(e.python_tid, (e.start_tid, e.device, e.resource));
    }
}

fn kind_of(call_type: u8) -> i32 {
    if call_type == PY_C_CALL {
        K_PY_C_CALL
    } else {
        K_PY_CALL
    }
}

/// Pairs enters with exits and pushes the resulting events into `arr`.
pub fn replay(
    cache: &mut PyCache,
    arr: &mut EventArray,
    enters: &mut [EnterRecord],
    exits: &[ExitRecord],
    end_time_ns: i64,
) {
    enters.sort_by_key(|e| e.start_ns);

    let mut out: Vec<Emitted> = Vec::new();
    replay_one_kind(cache, enters, exits, end_time_ns, K_PY_CALL, &mut out);
    replay_one_kind(cache, enters, exits, end_time_ns, K_PY_C_CALL, &mut out);
    out.sort_by_key(|e| e.start_ns);

    // Python ids number the events in the order they are materialized, so they
    // are handed out after the sort.
    for (n, e) in out.iter().enumerate() {
        let Some(view) = cache.site(e.site) else {
            continue;
        };
        let tag = if view.event_type == K_PY_C_CALL {
            EventType::PyCCall
        } else {
            EventType::PyCall
        };
        let i = arr.push(Event {
            tag_raw: tag as i32,
            start_time_ns: e.start_ns,
            start_tid: e.start_tid,
            device: e.device,
            resource: e.resource,
            end_time_ns: e.end_ns,
            python_tid: e.python_tid as usize,
            python_id: n + 1,
            parent: -1,
            linked: -1,
            ..Default::default()
        });
        // The strings live in the cache; copy them onto the event.
        let Some(slot) = arr.get_mut(i) else { continue };
        unsafe {
            if tag == EventType::PyCall {
                slot.callsite = event::FrameState {
                    line_no: view.callsite_line,
                    filename: owned(view.callsite_filename),
                    funcname: owned(view.callsite_name),
                };
                if view.has_module != 0 {
                    slot.module = Some(event::ModuleInfo {
                        cls_name: owned(view.module_cls_name),
                        id: view.module_id as usize,
                    });
                }
            } else {
                slot.function_name = owned(view.function_name);
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn eprof_pycache_replay(
    cache: *mut PyCache,
    arr: *mut EventArray,
    enters: *mut EnterRecord,
    n_enters: usize,
    exits: *const ExitRecord,
    n_exits: usize,
    end_time_ns: i64,
) {
    if cache.is_null() || arr.is_null() {
        return;
    }
    let enters = if enters.is_null() {
        &mut [][..]
    } else {
        std::slice::from_raw_parts_mut(enters, n_enters)
    };
    let exits = if exits.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(exits, n_exits)
    };
    replay(&mut *cache, &mut *arr, enters, exits, end_time_ns);
}

#[cfg(test)]
mod replay_tests {
    use super::*;
    use crate::cache::{CodeLoc, SiteKeyArgs, PY_CALL, PY_MODULE_CALL};
    use std::ffi::CString;

    fn cache_with_two_frames() -> PyCache {
        let mut c = PyCache::default();
        c.put_location(
            CodeLoc { filename: 1, name: 1, line: 10 },
            CString::new("/a.py").unwrap(),
            CString::new("outer").unwrap(),
        );
        c.put_location(
            CodeLoc { filename: 2, name: 2, line: 20 },
            CString::new("/a.py").unwrap(),
            CString::new("inner").unwrap(),
        );
        c
    }

    fn key(c: &mut PyCache, call_type: u8, value: CodeLoc, ptr: u64) -> u64 {
        c.intern(&SiteKeyArgs {
            call_type,
            python_tid: 0,
            value_loc: value,
            value_ptr: ptr,
            caller: CodeLoc { filename: 1, name: 1, line: 10 },
        })
    }

    fn enter(k: u64, t: i64) -> EnterRecord {
        EnterRecord { key: k, system_tid: 7, device: 0, resource: 0, start_ns: t }
    }

    #[test]
    fn an_exit_closes_the_frame_entered_most_recently() {
        let mut c = cache_with_two_frames();
        let outer = key(&mut c, PY_CALL, CodeLoc { filename: 1, name: 1, line: 10 }, 0);
        let inner = key(&mut c, PY_CALL, CodeLoc { filename: 2, name: 2, line: 20 }, 0);
        let mut arr = EventArray::default();
        // The third enter is what drives the drain -- see the test below.
        let mut enters = [enter(outer, 100), enter(inner, 200), enter(outer, 500)];
        let exits = [
            ExitRecord { t: 300, python_tid: 0, is_c_call: 0 },
            ExitRecord { t: 400, python_tid: 0, is_c_call: 0 },
        ];
        replay(&mut c, &mut arr, &mut enters, &exits, 1000);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(1).unwrap().end_time_ns, 300, "inner closes first");
        assert_eq!(arr.get(0).unwrap().end_time_ns, 400, "then outer");
    }

    #[test]
    fn exits_after_the_last_enter_are_left_to_the_end_of_the_trace() {
        let mut c = cache_with_two_frames();
        let k = key(&mut c, PY_CALL, CodeLoc { filename: 1, name: 1, line: 10 }, 0);
        let mut arr = EventArray::default();
        let mut enters = [enter(k, 100)];
        let exits = [ExitRecord { t: 200, python_tid: 0, is_c_call: 0 }];
        replay(&mut c, &mut arr, &mut enters, &exits, 1000);
        assert_eq!(arr.get(0).unwrap().end_time_ns, 1000);
    }

    #[test]
    fn a_frame_still_open_at_the_end_runs_to_the_end_of_the_trace() {
        let mut c = cache_with_two_frames();
        let k = key(&mut c, PY_CALL, CodeLoc { filename: 1, name: 1, line: 10 }, 0);
        let mut arr = EventArray::default();
        let mut enters = [enter(k, 100)];
        replay(&mut c, &mut arr, &mut enters, &[], 999);
        assert_eq!(arr.get(0).unwrap().end_time_ns, 999);
    }

    #[test]
    fn a_c_call_exit_does_not_close_a_python_frame() {
        // The two streams are replayed apart; sharing a stack would pair the
        // C exit with the python frame that happens to be open.
        let mut c = cache_with_two_frames();
        c.put_c_name(0xF00, CString::new("<built-in>").unwrap());
        let py = key(&mut c, PY_CALL, CodeLoc { filename: 1, name: 1, line: 10 }, 0);
        let cc = key(&mut c, PY_C_CALL, CodeLoc::default(), 0xF00);
        let mut arr = EventArray::default();
        let mut enters = [enter(py, 100), enter(cc, 200), enter(cc, 300)];
        let exits = [ExitRecord { t: 250, python_tid: 0, is_c_call: 1 }];
        replay(&mut c, &mut arr, &mut enters, &exits, 900);
        let py_ev = (0..arr.len()).map(|i| arr.get(i).unwrap())
            .find(|e| e.tag() == EventType::PyCall).unwrap();
        let c_ev = (0..arr.len()).map(|i| arr.get(i).unwrap())
            .find(|e| e.tag() == EventType::PyCCall && e.start_time_ns == 200).unwrap();
        assert_eq!(c_ev.end_time_ns, 250, "the C exit closes the C call");
        assert_eq!(py_ev.end_time_ns, 900, "and leaves the python frame open");
    }

    #[test]
    fn a_startup_frame_borrows_the_thread_of_the_next_event_on_its_python_tid() {
        let mut c = cache_with_two_frames();
        let startup = key(&mut c, PY_CALL, CodeLoc { filename: 1, name: 1, line: 10 }, 0);
        let normal = key(&mut c, PY_CALL, CodeLoc { filename: 2, name: 2, line: 20 }, 0);
        let mut arr = EventArray::default();
        let mut enters = [
            EnterRecord { key: startup, system_tid: NO_TID, device: 0, resource: 0, start_ns: 100 },
            EnterRecord { key: normal, system_tid: 42, device: 3, resource: 4, start_ns: 200 },
        ];
        replay(&mut c, &mut arr, &mut enters, &[], 900);
        let e = arr.get(0).unwrap();
        assert_eq!(e.start_tid, 42, "it was on the stack before we started");
        assert_eq!(e.device, 3);
        assert_eq!(e.resource, 4);
    }

    #[test]
    fn python_ids_number_the_events_in_start_order_from_one() {
        let mut c = cache_with_two_frames();
        let a = key(&mut c, PY_CALL, CodeLoc { filename: 1, name: 1, line: 10 }, 0);
        let b = key(&mut c, PY_CALL, CodeLoc { filename: 2, name: 2, line: 20 }, 0);
        let mut arr = EventArray::default();
        // Handed over out of order; the replay sorts them.
        let mut enters = [enter(b, 500), enter(a, 100)];
        replay(&mut c, &mut arr, &mut enters, &[], 900);
        assert_eq!(arr.get(0).unwrap().python_id, 1);
        assert_eq!(arr.get(0).unwrap().start_time_ns, 100);
        assert_eq!(arr.get(1).unwrap().python_id, 2);
    }

    #[test]
    fn an_enter_whose_key_was_never_interned_is_dropped() {
        let mut c = cache_with_two_frames();
        let mut arr = EventArray::default();
        let mut enters = [enter(9999, 100), enter(0, 200)];
        replay(&mut c, &mut arr, &mut enters, &[], 900);
        assert_eq!(arr.len(), 0);
    }
}
