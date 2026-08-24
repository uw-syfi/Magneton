#!/usr/bin/env python3
"""Build the environments the examples need, run them, and check what they wrote."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REEXEC = "AE_PY_REEXEC"


def _reexec_with_toml() -> None:
    """Re-run this script under a python that can read TOML, if this one cannot."""
    if os.environ.get(_REEXEC):
        return  # already came back here once; fall through and say so plainly
    uv = shutil.which("uv")
    if not uv:
        return
    found = subprocess.run(
        [uv, "python", "find", ">=3.11"], capture_output=True, text=True
    )
    if found.returncode:
        # Nothing suitable installed; uv puts one in its own directory.
        if subprocess.run([uv, "python", "install", "3.12"]).returncode:
            return
        found = subprocess.run(
            [uv, "python", "find", ">=3.11"], capture_output=True, text=True
        )
        if found.returncode:
            return
    interpreter = found.stdout.strip()
    if not interpreter or not Path(interpreter).exists():
        return
    os.execve(
        interpreter,
        [interpreter, str(Path(__file__).resolve()), *sys.argv[1:]],
        {**os.environ, _REEXEC: "1"},
    )


try:
    import tomllib
except ModuleNotFoundError:  # python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        _reexec_with_toml()  # does not return when it succeeds
        # Only reached when uv could not produce one. `doctor` still runs and
        # reports it; anything that reads the manifest stops in `load`.
        tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "examples" / "manifest.toml"
AE = ROOT / ".ae"
ENVS = AE / "envs"
LOGS = AE / "logs"
RESULTS = AE / "results.json"
CARGO_TARGETS = AE / "cargo-target"

CRATE = ROOT / "lib" / "eprof"


def build_requires() -> list[str]:
    """What every environment needs to build eprof, on top of its own pins."""
    if tomllib is None:
        sys.exit(f"reading {CRATE}/pyproject.toml needs a TOML parser; see `doctor`.")
    with (CRATE / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["build-system"]["requires"]


def sh(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **kw)


# --- The manifest ------------------------------------------------------------


@dataclass
class Env:
    name: str
    description: str
    python: str
    requires: list[str]

    @property
    def path(self) -> Path:
        return ENVS / self.name

    @property
    def interpreter(self) -> Path:
        return self.path / "bin" / "python"


@dataclass
class Example:
    name: str
    env: str
    dir: str
    script: str
    produces: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    about: str = ""
    known_issue: str = ""

    @property
    def path(self) -> Path:
        return ROOT / "examples" / self.dir


def load() -> tuple[dict[str, Env], list[Example]]:
    if tomllib is None:
        sys.exit(
            "Reading examples/manifest.toml needs a TOML parser: python 3.11 or "
            "newer, or `pip install tomli`. (`scripts/ae.py doctor` runs without one.)"
        )
    with MANIFEST.open("rb") as fh:
        raw = tomllib.load(fh)
    envs = {
        name: Env(name=name, **spec) for name, spec in raw.get("env", {}).items()
    }
    examples = [Example(**e) for e in raw.get("example", [])]
    for e in examples:
        if e.env not in envs:
            sys.exit(f"{e.name} names environment {e.env!r}, which the manifest does not define")
    return envs, examples


# --- doctor ------------------------------------------------------------------


def doctor() -> bool:
    """Report whether this machine has what the runs need."""
    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  {'PASS' if good else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        ok = ok and good

    print("Checking this machine:")

    gpu = shutil.which("nvidia-smi")
    names = ""
    if gpu:
        out = sh(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        names = ", ".join(out.stdout.split("\n")[:2]).strip(", ")
    check("an NVIDIA GPU", bool(gpu and names), names or "every example profiles CUDA work")
    check("uv", bool(shutil.which("uv")), "https://docs.astral.sh/uv/  (creates the environments)")
    check("cargo", bool(shutil.which("cargo")), "https://rustup.rs  (builds eprof)")

    clang = next((c for c in ("clang++-18", "clang++") if shutil.which(c)), None)
    version = ""
    if clang:
        first = sh([clang, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        version = first
    good_clang = bool(clang and "clang" in version.lower() and _major(version) >= 18)
    check("clang++ 18 or newer", good_clang, version or "needed for C++20 std::format")

    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    check("a CUDA toolkit", Path(cuda).is_dir(), cuda)

    header, libs, where = _cupti(cuda)
    check(
        "CUPTI",
        bool(header and libs),
        f"{header.parent}, {libs}" if header and libs
        else f"no {'cupti.h' if not header else 'libcupti'} under {where}; "
             f"install nvidia-cuda-cupti, or set EPROF_CUPTI_DIR",
    )

    check(
        "a TOML parser",
        tomllib is not None,
        "python 3.11+, or `pip install tomli`; needed to read examples/manifest.toml",
    )

    free_gb = shutil.disk_usage(ROOT).free // (1024**3)
    check("disk space", free_gb >= 60, f"{free_gb} GiB free; all the environments need roughly 60")

    print()
    print("OK." if ok else "Something above is missing; the runs that need it will fail.")
    return ok


def _cupti(cuda: str) -> tuple[Path | None, Path | None, str]:
    """CUPTI's header, its library directory, and where both were looked for."""
    override = os.environ.get("EPROF_CUPTI_DIR")
    roots = [Path(override)] if override else [Path(cuda) / "extras" / "CUPTI", Path(cuda)]
    where = override or cuda
    for root in roots:
        header = root / "include" / "cupti.h"
        if not header.is_file():
            continue
        for sub in ("lib64", "lib", f"targets/{platform.machine()}-linux/lib"):
            d = root / sub
            if d.is_dir() and any(d.glob("libcupti.so*")):
                return header, d, where
        return header, None, str(root)
    return None, None, where


def _major(version_line: str) -> int:
    try:
        return int(version_line.split("version")[1].strip().split()[0].split(".")[0])
    except Exception:
        return 0


# --- build -------------------------------------------------------------------


def build(env: Env, extra: list[str], force: bool = False) -> bool:
    """Creates the environment if it is not there, then builds eprof into it."""
    print(f"\n=== {env.name}: {env.description}")
    if force and env.path.exists():
        shutil.rmtree(env.path)

    if not env.interpreter.exists():
        print(f"  creating {env.path.relative_to(ROOT)} (python {env.python})")
        if sh(["uv", "venv", "--python", env.python, str(env.path)]).returncode:
            return fail(env.name, "could not create the virtualenv")

    wanted = env.requires + build_requires() + sorted(set(extra))
    print(f"  installing {len(wanted)} requirements")
    if sh(["uv", "pip", "install", "--python", str(env.interpreter), *wanted]).returncode:
        return fail(env.name, "the requirements do not resolve; see the output above")

    torch_version = installed_torch(env)
    if not torch_version:
        return fail(env.name, "torch is not importable after installing the requirements")
    print(f"  torch {torch_version}")

    if not shutil.which("cargo"):
        return fail(
            env.name,
            "cargo is not on PATH; part of the extension is Rust. "
            "Install it from https://rustup.rs and re-run.",
        )

    print("  building eprof against this environment's torch")
    target = CARGO_TARGETS / f"torch-{torch_version}"
    build_env = dict(
        os.environ,
        CARGO_TARGET_DIR=str(target),
        # The check above already reports a missing toolchain; maturin should
        # not quietly download one instead.
        MATURIN_NO_INSTALL_RUST="1",
        # build.rs asks this interpreter where torch and CUPTI are, so it can
        # only ever link the torch that will import it.
        PYO3_PYTHON=str(env.interpreter),
        # Build the extension from this checkout, and record it as such in the
        # wheel metadata rather than as coming from an index.
        EPROF_USE_PREBUILT="0",
    )
    started = time.time()
    proc = sh(
        ["uv", "pip", "install", "--python", str(env.interpreter), "--no-deps",
         "--no-build-isolation",
         "--reinstall-package", "magneton", "--reinstall-package", "magneton-eprof",
         str(ROOT / "python"), str(ROOT / "lib" / "eprof")],
        env=build_env,
    )
    if proc.returncode:
        return fail(env.name, "eprof did not build")
    print(f"  built in {time.time() - started:.0f}s")

    check = sh(
        [str(env.interpreter), "-c", VERIFY_LOCAL, str(ROOT)],
        capture_output=True, text=True,
    )
    print(check.stdout.strip() or check.stderr.strip())
    return check.returncode == 0


# Run inside a built environment: both packages import, and both came from this
# checkout rather than from an index, so the numbers are this tree's.
VERIFY_LOCAL = """
import json, sys
from importlib.metadata import distribution

import torch, magneton
from magneton import eprof

root = sys.argv[1]
for name in ("magneton", "magneton-eprof"):
    raw = distribution(name).read_text("direct_url.json")
    if raw is None:
        sys.exit(f"  {name} was installed from an index, not from {root}")
    url = json.loads(raw).get("url", "")
    if not url.startswith("file://") or root not in url:
        sys.exit(f"  {name} came from {url}, not from {root}")

print(f"  torch {torch.__version__}, magneton and its energy backend import,"
      f" both built from this checkout")
"""


def installed_torch(env: Env) -> str:
    """The torch this environment ended up with, which decides the build."""
    out = sh([str(env.interpreter), "-c", "import torch; print(torch.__version__)"],
             capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def fail(where: str, why: str) -> bool:
    print(f"  FAILED: {why}")
    return False


# --- run ---------------------------------------------------------------------


def run_example(ex: Example, env: Env, timeout: int) -> dict[str, Any]:
    """Runs one example and checks what it left behind."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{ex.name}.log"

    # Remove what it claims to produce, so a stale file from an earlier run
    # cannot be mistaken for this one's output.
    for name in ex.produces:
        (ex.path / name).unlink(missing_ok=True)

    started = time.time()
    with log.open("w") as fh:
        proc = subprocess.run(
            [str(env.interpreter), ex.script, *ex.args],
            cwd=ex.path,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONPATH=str(ROOT / "python")),
            timeout=None if timeout <= 0 else timeout,
            check=False,
        )
    elapsed = time.time() - started

    result: dict[str, Any] = {
        "name": ex.name,
        "env": ex.env,
        "script": f"{ex.dir}/{ex.script}",
        "seconds": round(elapsed, 1),
        "returncode": proc.returncode,
        "log": str(log.relative_to(ROOT)),
        "artifacts": {},
    }
    if proc.returncode != 0:
        # A failure the manifest already documents is reported as itself, not
        # as a surprise -- but it is still shown, and still not a pass.
        result["status"] = "known-issue" if ex.known_issue else "failed"
        result["why"] = (
            f"{ex.known_issue} (see {log.relative_to(ROOT)})" if ex.known_issue
            else f"exited {proc.returncode}; see {log.relative_to(ROOT)}"
        )
        return result

    problems = []
    for name in ex.produces:
        verdict = inspect_artifact(ex.path / name)
        result["artifacts"][name] = verdict
        if not verdict.startswith("ok"):
            problems.append(f"{name}: {verdict}")

    result["status"] = "passed" if not problems else "failed"
    if problems:
        result["why"] = "; ".join(problems)
    return result


def inspect_artifact(path: Path) -> str:
    """What an artifact is, or what is wrong with it."""
    if not path.exists():
        return "not written"
    size = path.stat().st_size
    if size == 0:
        return "empty"
    if path.suffix != ".json":
        return f"ok, {size:,} bytes"
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return f"not valid JSON ({e})"
    if isinstance(doc, dict) and "traceEvents" in doc:
        events = doc["traceEvents"]
        if not events:
            return "a trace with no events"
        power = sum(1 for e in events if e.get("name") == "[Power]" and "Power Usage" in e.get("args", {}))
        detail = f"ok, {len(events):,} events"
        return detail + (f", {power:,} power samples" if power else "")
    if isinstance(doc, dict) and "nodes" in doc:
        return f"ok, {len(doc['nodes'])} nodes, {len(doc.get('edges', []))} edges"
    return f"ok, {size:,} bytes"


# --- report ------------------------------------------------------------------


def report(results: list[dict[str, Any]]) -> int:
    if not results:
        print("Nothing has been run yet.")
        return 0

    width = max(len(r["name"]) for r in results)
    print(f"\n{'example'.ljust(width)}  {'env':10} {'status':7} {'time':>7}  detail")
    print("-" * (width + 46))
    for r in sorted(results, key=lambda r: (r["env"], r["name"])):
        detail = r.get("why", "")
        if not detail and r["artifacts"]:
            detail = "; ".join(f"{k} {v}" for k, v in list(r["artifacts"].items())[:2])
        print(
            f"{r['name'].ljust(width)}  {r['env']:10} {r['status']:7} "
            f"{r['seconds']:6.0f}s  {detail[:70]}"
        )

    passed = sum(1 for r in results if r["status"] == "passed")
    known = [r for r in results if r["status"] == "known-issue"]
    broken = [r for r in results if r["status"] not in ("passed", "known-issue")]

    print(f"\n{passed}/{len(results)} passed"
          + (f", {len(known)} known upstream issue{'s' if len(known) != 1 else ''}" if known else ""))
    for r in known:
        print(f"  known: {r['name']}: {r.get('why', '')}")
    for r in broken:
        print(f"  FAILED: {r['name']}: {r.get('why', 'failed')}")
    return 0 if not broken else 1


# --- entry point -------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stage", nargs="?", default="all",
                   choices=["all", "doctor", "build", "run", "report", "list"])
    p.add_argument("--env", action="append", help="only this environment (repeatable)")
    p.add_argument("--example", action="append", help="only this example (repeatable)")
    p.add_argument("--timeout", type=int, default=1800, help="per example, seconds; 0 for none")
    p.add_argument("--rebuild", action="store_true", help="discard and recreate the environments")
    p.add_argument("--skip-doctor", action="store_true",
                   help="run even if the machine checks fail")
    p.add_argument("--keep-going", action="store_true",
                   help="run the other environments even if one fails to build")
    args = p.parse_args()

    # Before the manifest, so that a machine missing the TOML parser is told so
    # by the check that exists to tell it rather than by `load` refusing.
    if args.stage == "doctor":
        return 0 if doctor() else 1

    envs, examples = load()

    if args.env:
        envs = {k: v for k, v in envs.items() if k in args.env}
        examples = [e for e in examples if e.env in envs]
    if args.example:
        examples = [e for e in examples if e.name in args.example]
        envs = {k: v for k, v in envs.items() if k in {e.env for e in examples}}

    if args.stage == "list":
        for name, env in envs.items():
            mine = [e for e in examples if e.env == name]
            print(f"\n{name}: {env.description}")
            print(f"  {', '.join(env.requires)}")
            for e in mine:
                print(f"    {e.name:26} {e.dir}/{e.script}")
        return 0

    if args.stage == "all" and not doctor() and not args.skip_doctor:
        # Stop rather than fail later, further from the cause.
        print("\nNothing above was run. Fix what failed, or --skip-doctor to try anyway.")
        return 1

    AE.mkdir(exist_ok=True)
    built: set[str] = set()

    if args.stage in ("all", "build"):
        for name, env in envs.items():
            extra = sorted({r for e in examples if e.env == name for r in e.requires})
            if build(env, extra, force=args.rebuild):
                built.add(name)
            elif not args.keep_going and args.stage == "build":
                return 1
        if args.stage == "build":
            print(f"\n{len(built)}/{len(envs)} environments ready")
            return 0 if len(built) == len(envs) else 1
    else:
        built = {name for name, env in envs.items() if env.interpreter.exists()}

    if args.stage in ("all", "run"):
        results: list[dict[str, Any]] = []
        if RESULTS.exists() and (args.env or args.example):
            # Keep what other selections already established.
            results = [
                r for r in json.loads(RESULTS.read_text())
                if r["name"] not in {e.name for e in examples}
            ]
        done: set[str] = {r["name"] for r in results if r["status"] == "passed"}

        for ex in examples:
            if ex.env not in built:
                print(f"  skipping {ex.name}: its environment did not build")
                results.append({
                    "name": ex.name, "env": ex.env, "script": f"{ex.dir}/{ex.script}",
                    "seconds": 0, "returncode": None, "status": "skipped",
                    "why": f"environment {ex.env} is not available", "artifacts": {},
                    "log": "",
                })
                continue
            missing = [n for n in ex.needs if n not in done]
            if missing:
                print(f"  skipping {ex.name}: needs {', '.join(missing)} to have run first")
                results.append({
                    "name": ex.name, "env": ex.env, "script": f"{ex.dir}/{ex.script}",
                    "seconds": 0, "returncode": None, "status": "skipped",
                    "why": f"needs {', '.join(missing)}", "artifacts": {}, "log": "",
                })
                continue

            print(f"  running {ex.name} ({ex.dir}/{ex.script})", flush=True)
            try:
                r = run_example(ex, envs[ex.env], args.timeout)
            except subprocess.TimeoutExpired:
                r = {
                    "name": ex.name, "env": ex.env, "script": f"{ex.dir}/{ex.script}",
                    "seconds": args.timeout, "returncode": None, "status": "failed",
                    "why": f"still running after {args.timeout}s", "artifacts": {},
                    "log": str((LOGS / f"{ex.name}.log").relative_to(ROOT)),
                }
            results.append(r)
            if r["status"] == "passed":
                done.add(ex.name)
            print(f"    {r['status']} in {r['seconds']:.0f}s"
                  + (f" -- {r.get('why', '')}" if r["status"] != "passed" else ""))

        RESULTS.write_text(json.dumps(results, indent=2))

    if args.stage in ("all", "run", "report"):
        stored = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
        return report(stored)

    return 0


if __name__ == "__main__":
    sys.exit(main())
