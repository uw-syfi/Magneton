//! GPU power sampling.
//!
//! A background thread polling NVML, and nothing else. It knows nothing about
//! events, traces or ops: it produces a series of power readings, and whoever
//! started it decides what those readings mean. That is why it is a crate of
//! its own -- nothing here depends on the profiler, and the profiler depends on
//! it only to start it, stop it, and take the series.

mod energy;

pub use energy::*;
