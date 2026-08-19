//! The three side streams a thread collects: power, allocations, OOMs.
//!
//! None of them come from a torch op. Power arrives from the sampler's thread,
//! allocations and OOMs from the caching allocator whenever it happens to run,
//! so they cannot be appended to the op stream and stay in order. Each gets its
//! own vector per thread, and all three are drained into the event array at the
//! end, where a single sort puts everything back on one timeline.
//!
//! They are plain data with a raw timestamp: the clock is only calibrated when
//! the run ends, so a stream stores what it read and converts on the way out.

/// One power reading, placed on a device rather than a thread.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PodPower {
    pub start_time: i64,
    pub power_usage_mw: i64,
    pub device_type: i8,
    pub device_index: i8,
}

/// One allocation or free -- a free is a negative `alloc_size`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PodAlloc {
    pub start_time: i64,
    pub ptr: u64,
    pub alloc_size: i64,
    pub total_allocated: u64,
    pub total_reserved: u64,
    pub device_type: i8,
    pub device_index: i8,
}

/// An allocation the caching allocator could not satisfy.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PodOom {
    pub start_time: i64,
    pub alloc_size: i64,
    pub total_allocated: u64,
    pub total_reserved: u64,
    pub device_type: i8,
    pub device_index: i8,
}

/// Per-subqueue POD event buffers.
impl PodEventStore {
    pub fn push_power(&mut self, e: PodPower) {
        self.power.push(e);
    }
    pub fn push_alloc(&mut self, e: PodAlloc) {
        self.alloc.push(e);
    }
    pub fn push_oom(&mut self, e: PodOom) {
        self.oom.push(e);
    }
}

#[derive(Default)]
pub struct PodEventStore {
    power: Vec<PodPower>,
    alloc: Vec<PodAlloc>,
    oom: Vec<PodOom>,
}

// Drain accessors: return a pointer + length to the internal buffer (valid until
// the next mutation / clear). The C++ side reads them at materialization.

// --- Drain ------------------------------------------------------------------

use eprof_storage::event::{Event, EventArray, EventType};

/// Converts a raw clock reading to trace nanoseconds. C++ owns the clock, so
/// it hands the conversion over as a callback rather than the numbers.
pub type TimeConverter = extern "C" fn(ctx: *mut std::ffi::c_void, t: i64) -> i64;

/// Moves every buffered sample onto the array and empties the store.
pub fn drain_into(
    store: &mut PodEventStore,
    arr: &mut EventArray,
    tid: u64,
    device: i32,
    resource: i32,
    convert: &dyn Fn(i64) -> i64,
) {
    let base = |tag: EventType, start: i64| Event {
        tag_raw: tag as i32,
        start_time_ns: convert(start),
        start_tid: tid,
        device,
        resource,
        parent: -1,
        linked: -1,
        ..Default::default()
    };

    for p in &store.power {
        arr.push(Event {
            power_usage_mw: p.power_usage_mw,
            device_type: p.device_type,
            device_index: p.device_index,
            ..base(EventType::Power, p.start_time)
        });
    }
    for a in &store.alloc {
        arr.push(Event {
            ptr: a.ptr,
            alloc_size: a.alloc_size,
            total_allocated: a.total_allocated,
            total_reserved: a.total_reserved,
            device_type: a.device_type,
            device_index: a.device_index,
            ..base(EventType::Allocation, a.start_time)
        });
    }
    for o in &store.oom {
        arr.push(Event {
            alloc_size: o.alloc_size,
            total_allocated: o.total_allocated,
            total_reserved: o.total_reserved,
            device_type: o.device_type,
            device_index: o.device_index,
            ..base(EventType::OutOfMemory, o.start_time)
        });
    }

    store.power.clear();
    store.alloc.clear();
    store.oom.clear();
}

#[cfg(test)]
mod drain_tests {
    use super::*;

    extern "C" fn double_it(_ctx: *mut std::ffi::c_void, t: i64) -> i64 {
        t * 2
    }

    #[test]
    fn draining_moves_every_stream_and_empties_the_store() {
        let mut store = PodEventStore::default();
        store.power.push(PodPower {
            start_time: 10,
            power_usage_mw: 250,
            device_type: 1,
            device_index: 3,
        });
        store.alloc.push(PodAlloc {
            start_time: 20,
            ptr: 0xBEEF,
            alloc_size: 512,
            total_allocated: 1024,
            total_reserved: 2048,
            device_type: 1,
            device_index: 3,
        });
        store.oom.push(PodOom {
            start_time: 30,
            alloc_size: 99,
            total_allocated: 1,
            total_reserved: 2,
            device_type: 1,
            device_index: 3,
        });
        let mut arr = EventArray::default();
        unsafe {
            drain_into(&mut store, &mut arr, 7, 4, 5, &|t| double_it(std::ptr::null_mut(), t));
        }

        assert_eq!(arr.len(), 3);
        let p = arr.get(0).unwrap();
        assert_eq!(p.tag(), EventType::Power);
        assert_eq!(p.start_time_ns, 20, "timestamps go through the converter");
        assert_eq!(p.power_usage_mw, 250);
        assert_eq!(p.start_tid, 7);
        assert_eq!((p.device, p.resource), (4, 5));
        assert_eq!(arr.get(1).unwrap().ptr, 0xBEEF);
        assert_eq!(arr.get(2).unwrap().tag(), EventType::OutOfMemory);
        assert_eq!(arr.get(2).unwrap().alloc_size, 99);
        assert_eq!(store.power.len() + store.alloc.len() + store.oom.len(), 0);
    }

    #[test]
    fn draining_an_empty_store_pushes_nothing() {
        let mut store = PodEventStore::default();
        let mut arr = EventArray::default();
        unsafe {
            drain_into(&mut store, &mut arr, 0, 0, 0, &|t| double_it(std::ptr::null_mut(), t));
        }
        assert_eq!(arr.len(), 0);
    }
}
