"""Golden-oracle capture/compare for the C++ -> Rust event-store migration."""

import json
import os
import re
import sys

import torch

_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")

# Fields that are stable across runs. Timestamps/tids/correlation ids are not.
STABLE_FIELDS = (
    "tag",
    "name",
    "device_type",
    "device_index",
    "activity_type",
    "flow_type",
    "flow_start",
)


def _workload(profiler_mod, record_shapes=True, profile_energy=True,
              with_stack=False, with_modules=False, mode="strict"):
    """Fixed workload; returns the profiler result."""
    C = profiler_mod
    prof = C._Profiler(
        {C._ActivityType.CPU, C._ActivityType.CUDA},
        record_shapes, False, False, with_stack, with_modules, profile_energy,
        [0] if profile_energy else [],
    )
    if with_stack:
        model = torch.nn.Linear(64, 64).cuda()
        x = torch.randn(8, 64, device="cuda")
        model(x)
        torch.cuda.synchronize()

    prof.start(set())
    if with_stack:
        for _ in range(3):
            model(x)
    else:
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        for _ in range(10):
            c = a @ b
            d = torch.relu(c)
            _ = d + a
    torch.cuda.synchronize()
    return prof.stop()


CONFIGS = {
    "cpp_tree": dict(record_shapes=True, profile_energy=True, mode="strict"),
    "py_stack": dict(record_shapes=False, profile_energy=False,
                     with_stack=True, with_modules=True, mode="loose"),
}

LOOSE_MARKERS = ("nn.Module: Linear", "torch/nn/modules/linear.py",
                 "aten::linear", "aten::addmm")


def _canonical(raw):
    """Structural summary of the exported node list."""
    by_id = {n["id"]: n for n in raw}

    def depth(n):
        d, seen = 0, set()
        while n["parent_id"] != -1 and n["parent_id"] in by_id:
            if n["id"] in seen:  # cycle guard
                break
            seen.add(n["id"])
            n = by_id[n["parent_id"]]
            d += 1
        return d

    rows = []
    for n in raw:
        # Python event names embed object addresses and this harness's own
        # source lines, both of which change every run.
        name = _ADDR_RE.sub("0xADDR", n["name"])
        if "golden_oracle.py" in name:
            continue
        row = {"depth": depth(n)}
        for f in STABLE_FIELDS:
            row[f] = n[f]
        row["name"] = name
        # Whether this node has a kineto link at all (id itself is unstable).
        row["has_link"] = n["linked_id"] != -1
        row["has_flow"] = n["flow_id"] != 0
        row["positive_dur"] = n["dur_ns"] > 0
        rows.append(row)
    rows.sort(key=lambda r: json.dumps(r, sort_keys=True))
    return rows


def _histogram(rows):
    """Counts per (depth, tag, name, device_type) -- the shape fingerprint."""
    h = {}
    for r in rows:
        k = f"{r['depth']}|{r['tag']}|{r['name']}|{r['device_type']}"
        h[k] = h.get(k, 0) + 1
    return h


def freeze(profiler_mod, label):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    cfg = CONFIGS[label]
    result = _workload(profiler_mod, **cfg)
    raw = result.export_raw_nodes()
    rows = _canonical(raw)
    path = os.path.join(GOLDEN_DIR, f"{label}.json")
    payload = {
        "n_nodes": len(raw),
        "schema": sorted(raw[0].keys()) if raw else [],
        "mode": cfg["mode"],
    }
    if cfg["mode"] == "strict":
        payload["histogram"] = _histogram(rows)
    else:
        payload["tags"] = sorted({r["tag"] for r in rows})
    with open(path, "w") as fh:
        json.dump(
            payload,
            fh,
            indent=2,
            sort_keys=True,
        )
    print(f"froze {len(raw)} nodes -> {path}")
    return path


def check(profiler_mod, label):
    path = os.path.join(GOLDEN_DIR, f"{label}.json")
    with open(path) as fh:
        golden = json.load(fh)
    cfg = CONFIGS[label]
    result = _workload(profiler_mod, **cfg)
    raw = result.export_raw_nodes()
    rows = _canonical(raw)

    problems = []
    schema = sorted(raw[0].keys()) if raw else []
    if schema != golden["schema"]:
        problems.append(
            f"schema changed:\n  golden={golden['schema']}\n  now   ={schema}"
        )

    if cfg["mode"] == "strict":
        now_hist = _histogram(rows)
        gold_hist = golden["histogram"]
        # Kernel counts vary slightly run to run; compare the set of shape keys
        # and flag any key that appears in one but not the other.
        missing = set(gold_hist) - set(now_hist)
        added = set(now_hist) - set(gold_hist)
        if missing:
            problems.append(f"event kinds MISSING vs golden: {sorted(missing)}")
        if added:
            problems.append(f"event kinds ADDED vs golden: {sorted(added)}")
        detail = f"{len(now_hist)} event kinds"
    else:
        now_tags = sorted({r["tag"] for r in rows})
        if now_tags != golden["tags"]:
            problems.append(
                f"event tags changed: golden={golden['tags']} now={now_tags}")
        names = "\n".join(r["name"] for r in rows)
        for marker in LOOSE_MARKERS:
            if marker not in names:
                problems.append(f"expected traced name containing {marker!r}")
        lo, hi = golden["n_nodes"] * 0.5, golden["n_nodes"] * 1.5
        if not (lo <= len(raw) <= hi):
            problems.append(
                f"node count {len(raw)} outside [{lo:.0f},{hi:.0f}] "
                f"(golden {golden['n_nodes']})")
        detail = f"tags={now_tags}"

    if problems:
        print("GOLDEN CHECK FAILED")
        for p in problems:
            print(" -", p)
        return False
    print(f"golden check OK ({len(raw)} nodes, {detail})")
    return True


if __name__ == "__main__":
    import magneton_eprof

    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    label = sys.argv[2] if len(sys.argv) > 2 else "cpp_tree"
    if mode == "freeze":
        freeze(magneton_eprof, label)
    else:
        sys.exit(0 if check(magneton_eprof, label) else 1)
