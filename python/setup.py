"""Where the energy extra's extension comes from.

`magneton[energy]` needs `magneton-eprof`, the compiled half. There are two
places it can come from, and which one is right depends on why you are
installing:

  * **A published wheel**, the default. Nothing to build, no CUDA toolkit, no
    clang. This is what `pip install magneton[energy]` should do for anyone who
    just wants to measure energy.

  * **This checkout**, when `EPROF_USE_PREBUILT=0`. Builds `lib/eprof` from
    source with maturin. For working on the extension, and for platforms or
    torch versions no published wheel matches -- the extension links the
    libtorch that will import it, so a prebuilt wheel is only usable against
    the torch it was built for.

        pip install './python[energy]'                      # published
        EPROF_USE_PREBUILT=0 pip install './python[energy]'  # from lib/eprof

The extras live here rather than in pyproject.toml because that switch has to
be read at build time, and a static table cannot branch. Everything else about
the distribution is still declared there.
"""

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
    # A direct reference, so the resulting metadata names this machine's path.
    # That is the point -- and the reason a wheel built this way should not be
    # handed to anyone else.
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
            # Pinned to dynamo's contemporary. transformers 4.57+ builds a
            # DynamicCache in GPT2's forward in a way torch 2.7's dynamo cannot
            # trace ("Unexpected type in sourceless builder builtins.method"),
            # which fails test_pytorch_141210_hf before any profiling happens.
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
