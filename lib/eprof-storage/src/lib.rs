//! Where a run accumulates.
//!
//! One array of events, and the tree built over it when the run ends. Every
//! part of the profiler writes here: the torch callbacks drain their per-thread
//! queues into it, the python tracer replays frames into it, kineto merges its
//! activities into it and reads back the torch ops to correlate against.
//!
//! Because all three write into the same array, this cannot live with the code
//! that drives them -- the dependency would run both ways. So it sits below all
//! of them and depends on nothing, which also means it builds and tests without
//! libtorch, CUDA or any C++ at all.

pub mod event;
pub mod frames;
pub mod materialize;
