//! Linking the extension module.
//!
//! Each subsystem builds and links its own C++ half, so nothing about those
//! archives is decided here. What is left is the shared libraries the module
//! must be bound to, and one link-order flag.
//!
//!   libtorch    including torch_python, since the interpreter loads this.
//!   libcupti    not on any default search path, and shipped by the nvidia
//!               wheels under a version suffix with no `.so` symlink, so it is
//!               resolved by file rather than by `-l`.
//!   libcudart   whichever the installed toolkit provides.
//!
//! Every path comes from `eprof_utils`, which asks the environment. See there for
//! why none of them is written down.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // libkineto is also inside torch, which exports it. Without this the
    // module's calls would resolve to torch's copy, which only initializes
    // CUPTI when torch.profiler runs -- so the trace would come back empty.
    println!("cargo:rustc-link-arg=-Wl,-Bsymbolic");

    let torch = eprof_utils::torch_lib_dir();
    // --no-as-needed on the two reached only through libkineto and the
    // registered callbacks: the linker sees no undefined symbol for either, and
    // would drop the DT_NEEDED entry.
    for lib in ["libtorch_cpu", "libtorch_cuda"] {
        link_arg(format!("-Wl,--no-as-needed,{}", find(&torch, lib)));
    }
    for lib in ["libc10_cuda", "libc10", "libtorch", "libtorch_python"] {
        link_arg(find(&torch, lib));
    }
    rpath(&torch);

    if let Some(cupti) = eprof_utils::cupti_dir().map(|d| d.join("lib")) {
        link_arg(find(&cupti, "libcupti"));
        if let Some(nvperf) = eprof_utils::find_lib(&cupti, "libnvperf_host") {
            link_arg(nvperf.display().to_string());
        }
        rpath(&cupti);
    }
    if let Some(cudart) = eprof_utils::cuda_lib_dir().and_then(|d| eprof_utils::find_lib(&d, "libcudart"))
    {
        link_arg(cudart.display().to_string());
        rpath(cudart.parent().expect("a file has a directory"));
    }

    for flag in ["-ldl", "-lpthread", "-lstdc++"] {
        link_arg(flag.to_string());
    }
}

fn link_arg(a: String) {
    println!("cargo:rustc-link-arg={a}");
}

fn rpath(dir: &std::path::Path) {
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
}

fn find(dir: &std::path::Path, stem: &str) -> String {
    eprof_utils::find_lib(dir, stem)
        .unwrap_or_else(|| panic!("{stem} not found in {}", dir.display()))
        .display()
        .to_string()
}
