"""Comparing two runs of the same computation.

Most of the issues in `examples/artifacts` have the same shape: something is
done two ways -- two implementations, two precisions, two versions of a library
-- and one is worse. Totals say by how much. They do not say *where*, and on a
model of any size that is the only question worth asking.

The dataflow matcher answers it. Given two recorded graphs it divides both at
the tensors they agree on and reports the regions between, so a stretch of one
implementation lines up with whatever the other replaced it with. Attaching the
per-operation energy to those regions turns the aggregate into a list of places,
largest difference first.

    from magneton import compare

    with eprof.Profiler(model=a, ..., dataflow_config=dataflow) as (pa, ma):
        ma(x)
    slow = compare.Run.of(pa, "eager")          # read it out before the next run

    with eprof.Profiler(model=b, ..., dataflow_config=dataflow) as (pb, mb):
        mb(x)
    fast = compare.Run.of(pb, "fused")

    print(compare.compare(slow, fast))

Two things this will not save you from.

**A result has to be read before the next run starts.** libkineto is
process-wide, and preparing a second trace reuses the buffers the first result
still points at. `Run.of` does that reading, which is why it is a separate step
rather than something `compare` does at the end.

**Both sides have to be warmed up the same way.** The dataflow plugin computes a
mean and standard deviation for every tensor it records, on a node's first
execution. Whichever side records its dataflow inside the measured region pays
for that in its own numbers -- on one comparison it was 77% of the total, and it
made the faster system look 8x slower. `example_inputs` handles this for a
wrapped model; anything else has to run once with the profiler stopped.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from magneton.backends.base import Cost
from magneton.matching import MatchConfig, match_graphs
from magneton.matching.subgraph import SubgraphMatch
from magneton.dataflow import DataflowDAG
import time

# A dataflow node and the cost of the work it did join on the annotation the
# graph transform left around it, which is `forward_` plus the node's own name.
_FRAME = "forward_"


@dataclass
class Run:
    """One side of a comparison: a recorded graph, and what it cost.

    Holds nothing that points back into whatever measured it, so it survives
    the next run -- which a profiler's own result does not.

    `per_node` is empty when nothing measured the run. That is a usable state,
    not a broken one: two graphs can be matched on structure and tensor values
    alone, which is what correctness checking is.
    """

    label: str
    dag: DataflowDAG
    per_node: Dict[str, Cost] = field(default_factory=dict)

    @property
    def has_energy(self) -> bool:
        """Whether any energy was attributable at all.

        NVML reports at about a kilohertz and the series needs two readings to
        integrate between, so a region of a few milliseconds can finish inside
        one sampling interval and come back with nothing. That is worth saying
        rather than printing zero, which reads as "used no energy".
        """
        return any(r.gpu_energy_j for r in self.per_node.values())

    @classmethod
    def of(cls, recorder, label: str) -> "Run":
        """A recording, as one side of a comparison.

        Equivalent to `recorder.run(label)`. Do this before the next run
        starts: a cost backend may hold process-wide state that the next
        capture reuses, and the energy one does.
        """
        return recorder.run(label)

    def save(self, prefix: str) -> None:
        """Writes this run out, costs included.

        For comparisons the two sides of which cannot share a process: two
        versions of a library, or one that needs a source patch applied between
        them. `DataflowDAG.save` keeps the graph and the tensors; the per-node
        costs are this class's own and would otherwise be lost, which would
        leave the comparison able to say where the two differ but not what the
        difference cost.
        """
        self.dag.save(f"{prefix}_dataflow.json", f"{prefix}_tensors.pt")
        with open(f"{prefix}_cost.json", "w") as fh:
            json.dump(
                {
                    "label": self.label,
                    "per_node": {
                        name: {
                            "num_calls": r.num_calls, "num_kernels": r.num_kernels,
                            "cpu_time_ns": r.cpu_time_ns, "gpu_time_ns": r.gpu_time_ns,
                            "gpu_energy_j": r.gpu_energy_j,
                        }
                        for name, r in self.per_node.items()
                    },
                },
                fh,
            )

    @classmethod
    def load(cls, prefix: str, label: Optional[str] = None) -> "Run":
        """A run saved by `save`. Costs come back if they were written."""
        dag = DataflowDAG.load(f"{prefix}_dataflow.json", f"{prefix}_tensors.pt")
        dag.build_edges()

        per_node: Dict[str, Cost] = {}
        cost_path = pathlib.Path(f"{prefix}_cost.json")
        if cost_path.exists():
            saved = json.loads(cost_path.read_text())
            label = label or saved.get("label")
            per_node = {
                name: Cost(**values)
                for name, values in saved.get("per_node", {}).items()
            }
        return cls(label=label or prefix, dag=dag, per_node=per_node)

    def cost(self, node_ids) -> Tuple[float, float]:
        """GPU nanoseconds and joules for a set of nodes."""
        gpu_ns = energy_j = 0.0
        for node_id in node_ids:
            node = self.dag.nodes.get(node_id)
            if node is None:
                continue
            # Two conventions, because two recorders fill this in. The FX
            # transform names a node `op_wrapper_N_...` and the scope around it
            # `forward_op_wrapper_N_...`, so the costs come back under the
            # second; a recorder that annotates with the node's own name -- the
            # jax one does -- comes back under the first. Trying both is
            # cheaper than making either lie about what it called things.
            record = self.per_node.get(_FRAME + node.node_name)
            if record is None:
                record = self.per_node.get(node.node_name)
            if record is not None:
                gpu_ns += record.gpu_time_ns
                energy_j += record.gpu_energy_j
        return gpu_ns, energy_j

    def describe(self, node_ids, limit: int = 3) -> str:
        """What a set of nodes computes, as the operations it performs."""
        seen: List[str] = []
        for node_id in sorted(node_ids):
            node = self.dag.nodes.get(node_id)
            if node is None:
                continue
            name = _readable(str(node.target))
            if name and name not in seen:
                seen.append(name)
            if len(seen) == limit:
                break
        return ", ".join(seen) or "?"


@dataclass
class Region:
    """One place the two runs agree about, and what each spent there."""

    match: SubgraphMatch
    a_gpu_ns: float
    a_energy_j: float
    b_gpu_ns: float
    b_energy_j: float
    what: str

    @property
    def sizes(self) -> Tuple[int, int]:
        return self.match.size()

    @property
    def energy_ratio(self) -> float:
        return self.b_energy_j / self.a_energy_j if self.a_energy_j else float("nan")

    @property
    def time_ratio(self) -> float:
        return self.b_gpu_ns / self.a_gpu_ns if self.a_gpu_ns else float("nan")


@dataclass
class ComparisonReport:
    """What the two runs have in common, and what it cost each of them."""

    a: str
    b: str
    regions: List[Region]
    a_nodes: int
    b_nodes: int
    covered_a: int
    covered_b: int
    totals: Tuple[float, float, float, float]  # a_gpu_ns, a_j, b_gpu_ns, b_j
    seconds: float = 0.0
    a_has_energy: bool = True
    b_has_energy: bool = True
    # Whether each side was measured at all -- which is not the same as a
    # matched region having cost, since the cuts can land on nodes that launch
    # no kernels and leave a fully measured run with an empty table.
    a_measured: bool = False
    b_measured: bool = False

    def __str__(self) -> str:
        return self.render()

    def render(self, limit: int = 20) -> str:
        lines = []
        lines.append(f"{self.a} vs {self.b}")
        lines.append(
            f"  {self.a_nodes} operations vs {self.b_nodes}; "
            f"{len(self.regions)} equivalent regions "
            f"covering {self.covered_a} and {self.covered_b} of them"
        )
        if not self.regions:
            lines.append(
                "  Nothing matched. Either the two runs share no tensor values, or"
                "\n  the tolerance is too tight -- see MatchConfig.stat_tolerance."
            )
            return "\n".join(lines)

        has_cost = any(r.a_gpu_ns or r.b_gpu_ns for r in self.regions)
        no_energy = [
            label for label, ok in ((self.a, self.a_has_energy), (self.b, self.b_has_energy))
            if not ok
        ]
        header = (
            f"    {'ops':>12}  {self.a[:18]:>19}  {self.b[:18]:>19}   what it computes"
            if has_cost else f"    {'ops':>12}   what it computes"
        )
        lines.append("")
        lines.append(header)
        if has_cost:
            lines.append(f"    {'a / b':>12}  {'GPU us / mJ':>19}  {'GPU us / mJ':>19}")

        # Biggest difference first: that is where an explanation is owed.
        ordered = sorted(
            self.regions,
            key=lambda r: -(r.a_energy_j - r.b_energy_j) if has_cost else -r.sizes[0],
        )
        for region in ordered[:limit]:
            size_a, size_b = region.sizes
            if has_cost:
                lines.append(
                    f"    {size_a:5d} / {size_b:4d}  "
                    f"{region.a_gpu_ns / 1e3:11.1f} / {region.a_energy_j * 1e3:5.2f}  "
                    f"{region.b_gpu_ns / 1e3:11.1f} / {region.b_energy_j * 1e3:5.2f}   "
                    f"{region.what}"
                )
            else:
                lines.append(f"    {size_a:5d} / {size_b:4d}   {region.what}")
        if len(ordered) > limit:
            lines.append(f"    ... and {len(ordered) - limit} more")

        if not has_cost and (self.a_measured or self.b_measured):
            lines.append(
                "\n    The run was measured, but no matched region contains any of the"
                "\n    work: the cuts landed on nodes that launch no kernels. On a graph"
                "\n    of a few nodes there is barely a spine to cut on, and the"
                "\n    per-operation table answers the question better than this does."
            )

        if has_cost:
            a_gpu, a_j, b_gpu, b_j = self.totals
            lines.append("")
            lines.append(
                f"    {'total':>12}  {a_gpu / 1e3:11.1f} / {a_j * 1e3:5.2f}  "
                f"{b_gpu / 1e3:11.1f} / {b_j * 1e3:5.2f}"
            )
            lines.append(
                "    Regions meet at their cut points, so a node can belong to two of"
                "\n    them and the rows do not add up to this."
            )
            if a_gpu and a_j and b_j:
                lines.append(
                    f"    Over what matched, {self.b} takes {b_gpu / a_gpu:.2f}x the GPU "
                    f"time and {b_j / a_j:.2f}x the energy of {self.a}."
                )
            elif a_gpu:
                lines.append(
                    f"    Over what matched, {self.b} takes {b_gpu / a_gpu:.2f}x the GPU "
                    f"time of {self.a}."
                )
            if (a_gpu and not b_gpu) or (b_gpu and not a_gpu):
                lines.append(
                    "\n    One side's matched region carries no GPU work while the other's"
                    "\n    does, so the two are not bracketing the same thing. That happens"
                    "\n    on graphs with only a few nodes, where there is barely a spine to"
                    "\n    cut on -- the per-operation table is the better tool there."
                )
            if no_energy:
                lines.append(
                    f"\n    No energy for {' and '.join(no_energy)}: the power sampler "
                    "reports at about a\n    kilohertz and needs two readings to integrate "
                    "between, so a region that\n    finishes inside one interval has none to "
                    "attribute. Measure more work."
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "a": self.a,
            "b": self.b,
            "operations": {"a": self.a_nodes, "b": self.b_nodes},
            "covered": {"a": self.covered_a, "b": self.covered_b},
            "seconds": round(self.seconds, 1),
            "totals": {
                "a_gpu_us": self.totals[0] / 1e3, "a_energy_mj": self.totals[1] * 1e3,
                "b_gpu_us": self.totals[2] / 1e3, "b_energy_mj": self.totals[3] * 1e3,
            },
            "regions": [
                {
                    "a_ops": r.sizes[0], "b_ops": r.sizes[1],
                    "a_gpu_us": r.a_gpu_ns / 1e3, "a_energy_mj": r.a_energy_j * 1e3,
                    "b_gpu_us": r.b_gpu_ns / 1e3, "b_energy_mj": r.b_energy_j * 1e3,
                    "computes": r.what,
                }
                for r in sorted(self.regions, key=lambda r: -(r.a_energy_j - r.b_energy_j))
            ],
        }


def compare(
    a: Run,
    b: Run,
    stat_tolerance: float = 1e-3,
    min_subgraph_size: int = 2,
    verbose: bool = False,
) -> ComparisonReport:
    """Finds what the two runs compute the same way, and what each spent on it.

    Args:
        a, b: the two sides, read out with `Run.of` or `Run.load`.
        stat_tolerance: how far two tensors may differ and still count as the
            same value. The default suits comparisons that differ only by
            floating-point association. A comparison that changes the
            arithmetic -- TF32 against FP32, say -- needs more room, or nothing
            will match.
        min_subgraph_size: ignore regions smaller than this.

    Returns:
        A `ComparisonReport`, which prints as a table.
    """

    started = time.time()
    matches = match_graphs(
        a.dag, b.dag,
        MatchConfig(
            stat_tolerance=stat_tolerance,
            min_subgraph_size=min_subgraph_size,
            verbose=verbose,
        ),
    )

    regions = []
    for match in matches:
        a_gpu, a_j = a.cost(match.graph1_nodes)
        b_gpu, b_j = b.cost(match.graph2_nodes)
        regions.append(
            Region(
                match=match, a_gpu_ns=a_gpu, a_energy_j=a_j,
                b_gpu_ns=b_gpu, b_energy_j=b_j,
                what=a.describe(match.graph1_nodes),
            )
        )

    # Totals over the union of the matched nodes. Summing the rows would count
    # a node twice wherever two regions meet.
    a_nodes = set().union(*(m.graph1_nodes for m in matches)) if matches else set()
    b_nodes = set().union(*(m.graph2_nodes for m in matches)) if matches else set()
    a_gpu, a_j = a.cost(a_nodes)
    b_gpu, b_j = b.cost(b_nodes)

    return ComparisonReport(
        a=a.label, b=b.label, regions=regions,
        a_nodes=len(a.dag.nodes), b_nodes=len(b.dag.nodes),
        covered_a=len(a_nodes), covered_b=len(b_nodes),
        totals=(a_gpu, a_j, b_gpu, b_j),
        seconds=time.time() - started,
        a_has_energy=a.has_energy,
        b_has_energy=b.has_energy,
        a_measured=any(c.gpu_time_ns for c in a.per_node.values()),
        b_measured=any(c.gpu_time_ns for c in b.per_node.values()),
    )


def _readable(target: str) -> str:
    """An FX target as something worth printing.

    `<built-in method addmm of type object at 0x7f...>` is the usual shape; the
    address is noise and the wrapper words are too.
    """
    target = target.split(" of type object")[0].strip("<>")
    for prefix in ("built-in function ", "built-in method ", "function "):
        target = target.replace(prefix, "")
    target = target.split(" at 0x")[0].strip()
    return "" if target in ("", "None") else target
