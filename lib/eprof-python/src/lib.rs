//! The python tracer's half that is not CPython.
//!
//! `cache` interns a callsite so tracing does the minimum per call; `replay`
//! pairs the enters and exits afterwards. C++ keeps only what has to touch
//! CPython -- pulling fields out of a frame and handing them over.

pub mod cache;
pub mod replay;

pub use cache::*;
pub use replay::*;
