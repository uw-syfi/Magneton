//! Compiles this crate's C++ half: the CPython tracer.
//!
//! It needs pybind11's headers as well as the interpreter's, because the frame
//! walk borrows pybind's GIL guards and its `handle`/`str` wrappers rather than
//! hand-rolling reference counting.

fn main() {
    println!("cargo:rerun-if-changed=src/tracer.cpp");
    println!("cargo:rerun-if-changed=include/tracer.h");
    println!("cargo:rerun-if-changed=include/rust_py_cache.h");

    let mut b = eprof_utils::build();
    if let Some(d) = eprof_utils::ask_python("import pybind11;print(pybind11.get_include())") {
        b.flag(format!("-isystem{}", d.display()));
    }
    b.file("src/tracer.cpp").compile("eprof_python_tracer_cpp");
}
