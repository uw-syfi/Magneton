"""Event-tree diff harness for the C++ -> Rust materialization migration (R3).

The plan for moving tree construction (record_queue.cpp's kineto correlation/
flow merge) to Rust is build-then-diff: reconstruct the event tree in Rust from
the same raw inputs the C++ side used, then assert it matches the C++ tree
*node-for-node* before deleting the C++ post-processing.

This module is that comparison layer. It reduces a tree to a canonical,
order-insensitive form (siblings sorted by a stable key, since child *order* is
incidental) and diffs two canonical forms field-by-field. Because both trees come
from the *same capture*, absolute timestamps are directly comparable.

`canonicalize` is duck-typed over any node exposing ``name`` (str), ``tag``,
``correlation_id``, ``start_time_ns``, ``duration_time_ns``, and ``children`` --
which both the C++ ``_ProfilerEvent`` and any future Rust node can satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class CanonNode:
    """A tree node reduced to the fields that must match across implementations."""

    depth: int
    tag: str            # short EventType name: TorchOp / Kineto / Power / ...
    name: str
    start_ns: int
    dur_ns: int
    correlation_id: int


def _short_tag(tag: Any) -> str:
    # Accept '_EventType.Kineto', 'EventType.Kineto', or 'Kineto'.
    return str(tag).split(".")[-1]


def _sibling_key(n) -> tuple:
    return (n.start_time_ns, n.duration_time_ns, n.correlation_id, n.name, _short_tag(n.tag))


def canonicalize(roots) -> List[CanonNode]:
    """Flatten a tree to a deterministic pre-order list of ``CanonNode``.

    Siblings are visited in a stable sorted order so that incidental child
    insertion-order differences between implementations don't show up as diffs;
    genuine structural or field differences still do.
    """
    out: List[CanonNode] = []

    def visit(node, depth: int):
        out.append(
            CanonNode(
                depth=depth,
                tag=_short_tag(node.tag),
                name=node.name,
                start_ns=node.start_time_ns,
                dur_ns=node.duration_time_ns,
                correlation_id=node.correlation_id,
            )
        )
        for child in sorted(node.children, key=_sibling_key):
            visit(child, depth + 1)

    for root in sorted(roots, key=_sibling_key):
        visit(root, 0)
    return out


# EventType tag int -> short name (must match eprof::data::EventType).
_EVENT_TYPE_TAGS = {
    0: "TorchOp",
    1: "Power",
    2: "Allocation",
    3: "OutOfMemory",
    4: "PyCall",
    5: "PyCCall",
    6: "Kineto",
}


class RawNode:
    """A node reconstructed from the flat POD bridge (export_raw_nodes / Rust).

    Duck-typed for ``canonicalize``; keeps the original dict in ``raw`` so the
    merge inputs (flow/linked/etc.) remain available.
    """

    __slots__ = ("name", "_tag", "start_time_ns", "duration_time_ns",
                 "correlation_id", "children", "raw")

    def __init__(self, d: dict):
        self.name = d["name"]
        self._tag = _EVENT_TYPE_TAGS.get(d["tag"], str(d["tag"]))
        self.start_time_ns = d["start_ns"]
        self.duration_time_ns = d["dur_ns"]
        self.correlation_id = d["correlation_id"]
        self.children: list = []
        self.raw = d

    @property
    def tag(self):
        return self._tag


def tree_from_raw(raw_nodes) -> List[RawNode]:
    """Rebuild a tree from flat POD nodes (``id``/``parent_id``).

    This is the schema both the C++ ``export_raw_nodes`` and the future Rust
    materialization emit, so the same reconstruction feeds the diff harness for
    either side.
    """
    nodes = [RawNode(d) for d in raw_nodes]
    by_id = {d["id"]: n for d, n in zip(raw_nodes, nodes)}
    roots: List[RawNode] = []
    for d, n in zip(raw_nodes, nodes):
        parent = d.get("parent_id", -1)
        if parent is None or parent < 0:
            roots.append(n)
        else:
            by_id[parent].children.append(n)
    return roots


def diff(ref: List[CanonNode], cand: List[CanonNode], max_report: int = 50) -> List[str]:
    """Return human-readable mismatches between a reference and candidate tree.

    Empty list means the trees are identical under canonicalization.
    """
    msgs: List[str] = []
    if len(ref) != len(cand):
        msgs.append(f"node count differs: ref={len(ref)} cand={len(cand)}")
    for i, (a, b) in enumerate(zip(ref, cand)):
        if a != b:
            msgs.append(f"[{i}] ref={a} != cand={b}")
            if len(msgs) >= max_report:
                msgs.append("... (truncated)")
                break
    return msgs


def trees_equal(ref_roots, cand_roots) -> bool:
    """Convenience: True iff two trees canonicalize identically."""
    return not diff(canonicalize(ref_roots), canonicalize(cand_roots))


def summarize(roots) -> dict:
    """Quick stats for logging (node count by tag, total nodes, depth)."""
    nodes = canonicalize(roots)
    by_tag: dict = {}
    for n in nodes:
        by_tag[n.tag] = by_tag.get(n.tag, 0) + 1
    return {
        "total": len(nodes),
        "by_tag": by_tag,
        "max_depth": max((n.depth for n in nodes), default=0),
    }
