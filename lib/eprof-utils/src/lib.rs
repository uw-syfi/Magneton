//! How the C++ halves are compiled.

use std::path::{Path, PathBuf};
use std::process::Command;

pub fn build() -> cc::Build {
    let mut b = cc::Build::new();
    b.compiler(compiler());
    b.cpp(true)
        .std("gnu++20")
        .flag("-fPIC")
        .warnings(true)
        .define("_GLIBCXX_USE_CXX11_ABI", "1")
        .define(
            if env("EPROF_TORCH_DEBUG").is_some() { "EPROF_UNUSED" } else { "NDEBUG" },
            None,
        )
        .define("USE_KINETO", None)
        .define("USE_DISTRIBUTED", None)
        .define("USE_C10D_GLOO", None)
        .define("USE_C10D_NCCL", None)
        .define("USE_RPC", None)
        .define("USE_TENSORPIPE", None);

    match device() {
        Device::Cuda => {
            b.define("USE_CUDA", None).define("LIBKINETO_NOROCTRACER", None);
        }
        Device::Rocm => {
            b.define("USE_ROCM", None).define("LIBKINETO_NOCUPTI", None);
        }
    }

    let lib = lib_dir();
    // Our own headers: a subsystem sees its siblings' C ABIs by name.
    for sub in ["eprof-torch", "eprof-python", "eprof-kineto"] {
        b.include(lib.join(sub).join("include"));
    }
    // Vendored libkineto, and the fmt it carries.
    let kineto = lib.join("eprof-kineto/libkineto");
    for d in ["include", "src", "third_party/fmt/include"] {
        b.flag(format!("-isystem{}", kineto.join(d).display()));
    }
    for d in python_include_dirs().into_iter().chain(torch_include_dirs()).chain(cuda_include_dirs()) {
        b.flag(format!("-isystem{}", d.display()));
    }
    b
}

/// The C++ compiler, which is not a free choice.
pub fn compiler() -> String {
    if let Some(cxx) = env("CXX") {
        require_clang_18(&cxx);
        return cxx;
    }
    for candidate in ["clang++-18", "clang++"] {
        if which(candidate).is_some() && clang_major(candidate).is_some_and(|v| v >= 18) {
            return candidate.into();
        }
    }
    panic!("no clang++ >= 18 found. Install one, or point CXX at it.")
}

fn require_clang_18(cxx: &str) {
    match clang_major(cxx) {
        Some(v) if v >= 18 => {}
        Some(v) => panic!("CXX={cxx} is clang {v}; 18 or newer is needed for std::format"),
        None => panic!("CXX={cxx} does not look like clang; 18 or newer is needed"),
    }
}

/// The clang major version, or None if it is not clang at all.
fn clang_major(cxx: &str) -> Option<u32> {
    let out = Command::new(cxx).arg("--version").output().ok()?;
    let first = String::from_utf8(out.stdout).ok()?.lines().next()?.to_string();
    if !first.to_ascii_lowercase().contains("clang") {
        return None;
    }
    first
        .split("version")
        .nth(1)?
        .split_whitespace()
        .next()?
        .split('.')
        .next()?
        .parse()
        .ok()
}

fn which(program: &str) -> Option<PathBuf> {
    let out = Command::new("which").arg(program).output().ok()?;
    out.status
        .success()
        .then(|| PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
}

// --- What is installed ------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cuda,
    Rocm,
}

/// Which GPU backend to build for.
pub fn device() -> Device {
    match env("EPROF_DEVICE").as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("cuda") => return Device::Cuda,
        Some("rocm") => return Device::Rocm,
        Some(other) => panic!("EPROF_DEVICE must be cuda or rocm, got {other:?}"),
        None => {}
    }
    if cuda_home().is_some() {
        Device::Cuda
    } else if Path::new("/opt/rocm").is_dir() {
        Device::Rocm
    } else {
        panic!(
            "found neither CUDA nor ROCm. Install one, or set EPROF_DEVICE to \
             pick which to build for."
        )
    }
}

pub fn lib_dir() -> PathBuf {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets this"));
    manifest
        .parent()
        .expect("a crate under lib/ has lib/ above it")
        .to_path_buf()
}

pub fn python() -> String {
    if let Some(p) = env("PYO3_PYTHON") {
        return p;
    }
    if let Some(venv) = env("VIRTUAL_ENV") {
        let p = PathBuf::from(venv).join("bin/python3");
        if p.is_file() {
            return p.display().to_string();
        }
    }
    "python3".into()
}

/// Runs a one-liner in that python and returns the directory it printed.
pub fn ask_python(code: &str) -> Option<PathBuf> {
    let out = Command::new(python()).args(["-c", code]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let dir = PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    dir.is_dir().then_some(dir)
}

/// Where a python package installed its files.
fn package_dir(module: &str, sub: &str) -> Option<PathBuf> {
    ask_python(&format!(
        "import os,{module} as m;print(os.path.join(os.path.dirname(m.__file__),'{sub}'))"
    ))
}

pub fn torch_dir() -> PathBuf {
    if let Some(d) = env("EPROF_TORCH_DIR") {
        return PathBuf::from(d);
    }
    ask_python("import os,torch;print(os.path.dirname(torch.__file__))").expect(
        "could not find torch. Set EPROF_TORCH_DIR, or make `import torch` work \
         for the python being built against.",
    )
}

pub fn torch_lib_dir() -> PathBuf {
    torch_dir().join("lib")
}

fn torch_include_dirs() -> Vec<PathBuf> {
    let inc = torch_dir().join("include");
    vec![inc.join("torch/csrc/api/include"), inc]
}

fn python_include_dirs() -> Vec<PathBuf> {
    ask_python("import sysconfig;print(sysconfig.get_paths()['include'])")
        .into_iter()
        .collect()
}

pub fn cupti_dir() -> Option<PathBuf> {
    if let Some(d) = env("EPROF_CUPTI_DIR") {
        return Some(PathBuf::from(d));
    }
    package_dir("nvidia.cuda_cupti", "").or_else(|| {
        let home = cuda_home()?;
        [home.join("extras/CUPTI"), home].into_iter().find(|d| d.join("include/cupti.h").is_file())
    })
}

/// Where a CUPTI installation keeps its shared libraries.
pub fn cupti_lib_dir(cupti: &Path) -> Option<PathBuf> {
    [
        cupti.join("lib64"),
        cupti.join("lib"),
        cupti.join(format!("targets/{}-linux/lib", std::env::consts::ARCH)),
    ]
    .into_iter()
    .find(|d| find_lib(d, "libcupti").is_some())
}

pub fn cuda_home() -> Option<PathBuf> {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(d) = env(var) {
            return Some(PathBuf::from(d));
        }
    }
    if let Ok(out) = Command::new("which").arg("nvcc").output() {
        if out.status.success() {
            let nvcc = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
            if let Some(home) = nvcc.parent().and_then(Path::parent) {
                return Some(home.to_path_buf());
            }
        }
    }
    let conventional = PathBuf::from("/usr/local/cuda");
    conventional.is_dir().then_some(conventional)
}

/// Where the CUDA runtime and its friends live.
pub fn cuda_lib_dir() -> Option<PathBuf> {
    let home = cuda_home()?;
    [
        home.join(format!("targets/{}-linux/lib", std::env::consts::ARCH)),
        home.join("lib64"),
        home.join("lib"),
    ]
    .into_iter()
    .find(|d| d.is_dir())
}

fn cuda_include_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(home) = cuda_home() {
        for d in [
            home.join("include"),
            home.join(format!("targets/{}-linux/include", std::env::consts::ARCH)),
        ] {
            if d.is_dir() {
                out.push(d);
            }
        }
    }
    if let Some(c) = cupti_dir() {
        let inc = c.join("include");
        if inc.is_dir() {
            out.push(inc);
        }
    }
    out
}

// --- Resolving a shared library --------------------------------------------

/// Finds a library by stem, accepting a version suffix.
pub fn find_lib(dir: &Path, stem: &str) -> Option<PathBuf> {
    let plain = dir.join(format!("{stem}.so"));
    if plain.is_file() {
        return Some(plain);
    }
    let prefix = format!("{stem}.so.");
    let mut versioned: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&prefix))
        })
        .collect();
    versioned.sort();
    versioned.pop()
}

pub fn env(k: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={k}");
    std::env::var(k).ok().filter(|v| !v.is_empty())
}
