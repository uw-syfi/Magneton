//! One recorded entry into a python frame.
//!
//! Tracing splits an entry from its exit: the tracer stamps this while python
//! runs, and the pairing happens afterwards, so the traced program pays as
//! little as possible. Three places touch the record on that path -- the
//! tracer writes it, the per-thread queue carries it, the replay consumes it --
//! and they are in two different crates, so it belongs to neither.

/// One recorded entry into a frame.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EnterRecord {
    pub key: u64,
    /// The OS thread that recorded it, or `NO_TID` for a frame that was
    /// already on the stack when profiling started.
    pub system_tid: u64,
    pub device: i32,
    pub resource: i32,
    pub start_ns: i64,
}

/// A frame that was already on the stack when profiling started has no thread
/// of its own on record. The replay recognises them by this.
pub const NO_TID: u64 = u64::MAX;
