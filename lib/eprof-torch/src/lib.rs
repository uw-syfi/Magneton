//! Collecting what torch does: the RecordFunction callbacks, and the per-thread
//! stores they write into.
//!
//! torch enters here around every dispatched operator, on whatever thread is
//! running it, while the workload is being measured. That sets the shape of
//! everything below: the callbacks read their arguments and append, the stores
//! are per-thread so nothing contends, and every question that can wait until
//! the run ends does.
//!
//! `src/*.cpp` holds the reads that need a libtorch type in hand -- an
//! `at::RecordFunction`, a `c10::IValue`, the state torch reports allocations
//! into. Each is coarse: one crossing per op, not one per field.

pub mod inputs;
pub mod ops;
pub mod pods;
pub mod queue;
pub mod run;
