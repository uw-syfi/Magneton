//! Linking the extension module.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rustc-link-arg=-Wl,-Bsymbolic");

    let torch = eprof_utils::torch_lib_dir();
    for lib in ["libtorch_cpu", "libtorch_cuda"] {
        link_arg(format!("-Wl,--no-as-needed,{}", find(&torch, lib)));
    }
    for lib in ["libc10_cuda", "libc10", "libtorch", "libtorch_python"] {
        link_arg(find(&torch, lib));
    }
    rpath(&torch);

    if let Some(cupti) =
        eprof_utils::cupti_dir().as_deref().and_then(eprof_utils::cupti_lib_dir)
    {
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
