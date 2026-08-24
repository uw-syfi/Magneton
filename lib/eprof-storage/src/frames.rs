//! One recorded entry into a python frame.

/// One recorded entry into a frame.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EnterRecord {
    pub key: u64,
    pub system_tid: u64,
    pub device: i32,
    pub resource: i32,
    pub start_ns: i64,
}

pub const NO_TID: u64 = u64::MAX;
