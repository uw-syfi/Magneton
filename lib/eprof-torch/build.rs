//! Compiles this crate's C++ half.
//!
//! The four files here are the reads that need a libtorch type in hand. They
//! are part of this subsystem, so this crate builds them rather than leaving
//! them to a separate step that has to be remembered.

fn main() {
    let mut b = eprof_utils::build();
    for f in ["capture", "driver", "metadata", "state"] {
        let src = format!("src/{f}.cpp");
        println!("cargo:rerun-if-changed={src}");
        b.file(src);
    }
    for h in ["capture", "macros", "metadata", "state", "rust_op_inputs", "rust_queue", "rust_run"] {
        println!("cargo:rerun-if-changed=include/{h}.h");
    }
    b.compile("eprof_torch_tracer_cpp");
}
