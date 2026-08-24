"""Where the `energy` extra's extension comes from."""

import os
import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).resolve().parent
CRATE = HERE.parent / "lib" / "eprof"

# Anything falsey means "build it from this tree"; unset means "published".
_PREBUILT = os.environ.get("EPROF_USE_PREBUILT", "1").strip().lower()
USE_PREBUILT = _PREBUILT not in {"0", "false", "no", "off"}

if USE_PREBUILT:
    ENERGY = ["magneton-eprof"]
else:
    if not (CRATE / "Cargo.toml").exists():
        raise RuntimeError(
            f"EPROF_USE_PREBUILT=0 asks for the extension to be built from "
            f"{CRATE}, which is not there. That path only exists in a checkout; "
            f"unset the variable to install the published wheel instead."
        )
    # A direct reference, so the metadata names this machine's path; a wheel
    # built this way is not portable.
    ENERGY = [f"magneton-eprof @ {CRATE.as_uri()}"]

setup(
    extras_require={
        "energy": ENERGY,
        "dev": [
            "black==25.1.0",
            "yapf>=0.40.0",
            "ruff>=0.6.0",
            "codespell>=2.3.0",
            "clang-format>=14.0.0",
        ],
        "test": [
            "pytest",
            "transformers==4.51.3",
            "peft",
            "accelerate",
            "jax[cuda12]==0.5.0",
            "tensorflow==2.20.0",
            "megatron-core[dev]==0.12.0",
            "diffusers @ git+https://github.com/huggingface/diffusers.git@0454fbb",
            "vllm==0.9.1",
        ],
    },
)
