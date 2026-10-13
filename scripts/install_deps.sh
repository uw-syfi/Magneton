#!/usr/bin/env bash
# Check for, and install, everything outside Python that building Magneton's
# energy extension needs: the compilers, the CUDA toolkit, Rust and uv.
#
#   scripts/install_deps.sh            check, then install whatever is missing
#   scripts/install_deps.sh --check    only report
#
# Installing targets Ubuntu 22.04 and 24.04 and needs root (or sudo). Anything
# already present is left alone. Elsewhere, --check still says what is missing.
#
# Python packages are not installed here: scripts/ae.py creates its own
# environments, and the README's manual install puts them in a virtualenv.

set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-12.8}"   # the CUDA that torch 2.8's default wheels use
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

if [[ $(id -u) -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi
export DEBIAN_FRONTEND=noninteractive

# rustup and uv install into the home directory, and reach PATH only in shells
# that source their env files: a new terminal, or `docker exec` into a
# container, then has no cargo, and the build fails far from the cause. Both are
# therefore linked into /usr/local/bin, and checked for on the plain PATH.

# --- checks -------------------------------------------------------------------

clang_ok() {
    local cxx v
    for cxx in clang++-18 clang++; do
        command -v "$cxx" >/dev/null || continue
        v=$("$cxx" --version | head -1 | sed -E 's/.*version ([0-9]+).*/\1/')
        [[ "$v" =~ ^[0-9]+$ && "$v" -ge 18 ]] && return 0
    done
    return 1
}

# clang uses the newest libstdc++ it finds, and std::format needs GCC 13's.
libstdcxx_ok() {
    local d
    for d in /usr/include/c++/*; do
        [[ "${d##*/}" =~ ^[0-9]+$ && "${d##*/}" -ge 13 ]] && return 0
    done
    return 1
}
boost_ok() { [[ -f /usr/include/boost/stacktrace.hpp ]]; }
python_headers_ok() { ls /usr/include/python3*/Python.h >/dev/null 2>&1; }

cuda_home() {
    local d
    for d in "${CUDA_HOME:-}" "${CUDA_PATH:-}"; do [[ -n "$d" && -d "$d" ]] && { echo "$d"; return; }; done
    if command -v nvcc >/dev/null; then dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")"; return; fi
    [[ -d /usr/local/cuda ]] && echo /usr/local/cuda
}
cuda_ok() {
    local h; h=$(cuda_home) || true
    [[ -n "$h" ]] && { [[ -f "$h/extras/CUPTI/include/cupti.h" ]] || [[ -f "$h/include/cupti.h" ]]; }
}

gpu_ok() { command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; }
cargo_ok() { command -v cargo >/dev/null; }
uv_ok() { command -v uv >/dev/null; }

report() {
    local missing=0 name
    echo "Checking this machine:"
    for name in gpu clang libstdcxx boost python_headers cuda cargo uv; do
        local label
        case $name in
            gpu)            label="NVIDIA GPU and driver (nvidia-smi)" ;;
            clang)          label="clang++ 18 or newer" ;;
            libstdcxx)      label="libstdc++ 13 or newer (g++-13)" ;;
            boost)          label="boost headers" ;;
            python_headers) label="Python development headers" ;;
            cuda)           label="CUDA toolkit with CUPTI" ;;
            cargo)          label="Rust (cargo)" ;;
            uv)             label="uv" ;;
        esac
        if "${name}_ok"; then echo "  PASS  $label"; else echo "  FAIL  $label"; missing=1; fi
    done
    return $missing
}

# --- installs -----------------------------------------------------------------

apt_install() { $SUDO apt-get install -y --no-install-recommends "$@"; }

ubuntu_codename() { . /etc/os-release; echo "${VERSION_CODENAME:-}"; }

install_base() {
    $SUDO apt-get update
    apt_install ca-certificates curl wget gnupg git build-essential pkg-config \
        software-properties-common lsb-release
}

install_clang() {
    local codename; codename=$(ubuntu_codename)
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | $SUDO tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null
    echo "deb http://apt.llvm.org/$codename/ llvm-toolchain-$codename-18 main" \
        | $SUDO tee /etc/apt/sources.list.d/llvm-18.list >/dev/null
    $SUDO apt-get update
    apt_install clang-18
}

install_libstdcxx() {
    # 24.04 carries GCC 13; 22.04 needs the toolchain PPA.
    if ! apt-cache show libstdc++-13-dev >/dev/null 2>&1; then
        $SUDO add-apt-repository -y ppa:ubuntu-toolchain-r/test
        $SUDO apt-get update
    fi
    apt_install g++-13 libstdc++-13-dev
}

install_cuda() {
    local id ver arch pkg
    . /etc/os-release
    id="ubuntu${VERSION_ID//./}"
    arch=$(uname -m); [[ "$arch" == aarch64 ]] && arch=sbsa
    wget -qO /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/$id/$arch/cuda-keyring_1.1-1_all.deb"
    $SUDO dpkg -i /tmp/cuda-keyring.deb
    $SUDO apt-get update
    ver=${CUDA_VERSION/./-}
    # The toolkit without the driver: a container gets that from its host.
    apt_install "cuda-toolkit-$ver"
    [[ -e /usr/local/cuda ]] || $SUDO ln -s "/usr/local/cuda-$CUDA_VERSION" /usr/local/cuda
}

install_rust() {
    [[ -x "$HOME/.cargo/bin/cargo" ]] ||
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    local b; for b in cargo rustc rustup; do $SUDO ln -sf "$HOME/.cargo/bin/$b" /usr/local/bin/$b; done
}

install_uv() {
    [[ -x "$HOME/.local/bin/uv" ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
    local b; for b in uv uvx; do $SUDO ln -sf "$HOME/.local/bin/$b" /usr/local/bin/$b; done
}

# --- main ---------------------------------------------------------------------

if report; then
    echo; echo "Everything is in place."
    exit 0
fi
if [[ $CHECK_ONLY -eq 1 ]]; then
    echo; echo "Run scripts/install_deps.sh without --check to install what is missing."
    exit 1
fi
if ! command -v apt-get >/dev/null; then
    echo; echo "Installing needs apt (Ubuntu 22.04 or 24.04); install the missing pieces by hand."
    exit 1
fi

echo; echo "Installing what is missing."
install_base
clang_ok          || install_clang
libstdcxx_ok      || install_libstdcxx
boost_ok          || apt_install libboost-dev
python_headers_ok || apt_install python3-dev
cuda_ok           || install_cuda
cargo_ok          || install_rust
uv_ok             || install_uv

echo
if report; then
    echo; echo "Everything is in place."
    exit 0
fi
if ! gpu_ok; then
    echo
    echo "No GPU is visible. The driver cannot be installed from here: on a host,"
    echo "install the NVIDIA driver; in a container, start it with --gpus all."
fi
exit 1
