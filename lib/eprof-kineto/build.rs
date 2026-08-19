//! Builds this crate's C++: vendored libkineto, and the one libtorch lookup.
//!
//! libkineto is a CMake project of its own, so it is driven as one rather than
//! recompiled by hand -- it detects CUPTI, generates its source list from a
//! bazel file, and carries its own fmt.
//!
//! It is whole-archived: CUPTI callbacks are registered by static initializers,
//! so objects that nothing references by name still have to reach the final
//! link, or no GPU activity is ever collected.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/backend.cpp");
    println!("cargo:rerun-if-changed=include/backend.h");
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=libkineto/CMakeLists.txt");
    println!("cargo:rerun-if-changed=libkineto/src");
    println!("cargo:rerun-if-changed=libkineto/include");

    let kineto = build_libkineto();
    println!("cargo:rustc-link-search=native={}", kineto.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=kineto");
    // Downstream build scripts get this as DEP_KINETO_ROOT (see `links` in
    // Cargo.toml), which is how the module's link step finds the archive
    // without being told where the build happened.
    println!("cargo:root={}", kineto.display());

    eprof_utils::build().file("src/backend.cpp").compile("eprof_kineto_cpp");
}

/// Configures and builds vendored libkineto, returning the directory holding
/// `libkineto.a`. Everything below cmake's control lands in OUT_DIR, so a
/// `cargo clean` takes it with everything else.
fn build_libkineto() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo = eprof_utils::lib_dir().parent().expect("lib/ has a repo above it").to_path_buf();
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets this")).join("libkineto");
    std::fs::create_dir_all(&out).expect("could not create the libkineto build directory");

    let mut cfg = vec![
        format!("-DCMAKE_CXX_COMPILER={}", eprof_utils::compiler()),
        "-DCMAKE_BUILD_TYPE=Release".into(),
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON".into(),
        "-DKINETO_LIBRARY_TYPE=static".into(),
        // Its tests want gtest, which is not ours to provide.
        "-DKINETO_BUILD_TESTS=OFF".into(),
        format!("-DEPROF_THIRD_PARTY={}", repo.join("third_party").display()),
    ];
    match eprof_utils::device() {
        eprof_utils::Device::Cuda => {
            let cupti = eprof_utils::cupti_dir().expect(
                "CUDA is installed but CUPTI is not. Install nvidia-cuda-cupti, or \
                 set EPROF_CUPTI_DIR.",
            );
            let lib = cupti.join("lib");
            cfg.push("-DLIBKINETO_NOROCTRACER=ON".into());
            cfg.push("-DLIBKINETO_NOCUPTI=OFF".into());
            cfg.push(format!("-DCUDA_SOURCE_DIR={}", cuda_home().display()));
            cfg.push(format!("-DCUPTI_INCLUDE_DIR={}", cupti.join("include").display()));
            for (var, stem) in [
                ("CUDA_cupti_LIBRARY", "libcupti"),
                ("CUDA_nvperf_host_LIBRARY", "libnvperf_host"),
            ] {
                let found = eprof_utils::find_lib(&lib, stem)
                    .unwrap_or_else(|| panic!("{stem} not found in {}", lib.display()));
                cfg.push(format!("-D{var}={}", found.display()));
            }
        }
        eprof_utils::Device::Rocm => {
            cfg.push("-DLIBKINETO_NOROCTRACER=OFF".into());
            cfg.push("-DLIBKINETO_NOCUPTI=ON".into());
        }
    }

    let mut args = vec![
        "-S".to_string(),
        manifest.display().to_string(),
        "-B".to_string(),
        out.display().to_string(),
    ];
    args.extend(cfg);
    run("cmake", &args.iter().map(String::as_str).collect::<Vec<_>>());
    let jobs = std::thread::available_parallelism().map_or(1, |n| n.get()).to_string();
    run("cmake", &["--build", &out.display().to_string(), "--target", "kineto", "-j", &jobs]);

    // add_subdirectory puts the archive under a binary directory named for the
    // subdirectory, so look for it rather than predicting where cmake put it.
    find_archive(&out).unwrap_or_else(|| {
        panic!("cmake reported success but libkineto.a is not under {}", out.display())
    })
}

fn find_archive(dir: &std::path::Path) -> Option<PathBuf> {
    if dir.join("libkineto.a").is_file() {
        return Some(dir.to_path_buf());
    }
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .find_map(|p| find_archive(&p))
}

fn cuda_home() -> PathBuf {
    eprof_utils::cuda_home().expect("CUDA selected but no toolkit found; set CUDA_HOME")
}

fn run(program: &str, args: &[&str]) {
    let status = std::process::Command::new(program)
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("could not run {program}: {e}"));
    assert!(status.success(), "{program} {} failed", args.join(" "));
}
