"""The magneton_eprof stub has to keep up with the module."""

import ast
import pathlib

import pytest

import magneton_eprof

STUB = pathlib.Path(__file__).resolve().parents[2] / "lib" / "eprof" / "magneton_eprof.pyi"


def _stub() -> ast.Module:
    if not STUB.exists():
        pytest.skip(f"no stub at {STUB} (running against an installed wheel?)")
    return ast.parse(STUB.read_text())


def _public(names) -> set:
    return {n for n in names if not n.startswith("__")}


def test_stub_declares_every_module_export():
    declared = {
        n.name
        for n in _stub().body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    }
    # `magneton_eprof` itself is the submodule maturin's __init__ star-imports from, not
    # something the module means to export.
    exported = _public(dir(magneton_eprof)) - {"magneton_eprof"}

    assert not exported - declared, f"not in the stub: {sorted(exported - declared)}"
    assert not declared - exported, f"in the stub but not the module: {sorted(declared - exported)}"


def test_stub_classes_declare_every_member():
    for node in _stub().body:
        if not isinstance(node, ast.ClassDef):
            continue
        cls = getattr(magneton_eprof, node.name)
        declared = {
            b.name for b in node.body if isinstance(b, ast.FunctionDef)
        } | {
            t.target.id for t in node.body if isinstance(t, ast.AnnAssign)
        }
        declared = _public(declared)
        actual = _public(dir(cls))

        assert not actual - declared, (
            f"{node.name}: not in the stub: {sorted(actual - declared)}"
        )
        assert not declared - actual, (
            f"{node.name}: in the stub but not on the class: "
            f"{sorted(declared - actual)}"
        )


def test_the_stub_ships_with_the_module():
    """PEP 561: a type checker only looks."""
    installed = pathlib.Path(magneton_eprof.__file__).parent
    assert (installed / "py.typed").is_file(), "py.typed missing from the wheel"
    assert (installed / "__init__.pyi").is_file(), "the stub did not ship"
