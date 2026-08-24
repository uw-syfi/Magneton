//! Compiles this crate's C++ half.

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
