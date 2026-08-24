import csv
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

BUGGY_TRACE = os.path.join(HERE, "uneven_gpt.json")
NORMAL_TRACE = os.path.join(HERE, "normal_gpt.json")
OUTPUT_CSV = os.path.join(HERE, "study_gpt.csv")

GRID_US = 10000
WINDOW_US = 20000


def load_power_series(trace_path: str) -> pd.Series:
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    samples = [
        (e["ts"], e["args"]["Power Usage"])
        for e in events
        if e.get("name") == "[Power]" and e["ts"] > 0
    ]
    if not samples:
        raise RuntimeError(f"{trace_path} has no power samples with a timestamp")
    samples.sort(key=lambda x: x[0])
    t0 = samples[0][0]
    ts = np.array([t - t0 for t, _ in samples], dtype=float)
    p = np.array([p for _, p in samples], dtype=float)
    idx = pd.to_timedelta(ts, unit="us")
    s = pd.Series(p, index=idx).groupby(level=0).mean()
    return s


def resample_and_smooth(s: pd.Series) -> List[Tuple[float, float]]:
    grid = pd.timedelta_range(start=0, end=s.index.max(), freq=f"{GRID_US}us")
    s_grid = (
        s.reindex(s.index.union(grid)).interpolate(method="time").reindex(grid)
    )
    s_smooth = s_grid.rolling(f"{WINDOW_US}us", min_periods=1, center=True).mean()
    ts_us = (s_smooth.index.total_seconds() * 1e6).tolist()
    return list(zip(ts_us, s_smooth.values.tolist()))


def write_csv(
    path: str,
    buggy: List[Tuple[float, float]],
    normal: List[Tuple[float, float]],
) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "power(buggy)", "timestamp", "power(normal)"])
        for i in range(max(len(buggy), len(normal))):
            row = []
            if i < len(buggy):
                row += [f"{buggy[i][0]:.3f}", f"{buggy[i][1]:.3f}"]
            else:
                row += ["", ""]
            if i < len(normal):
                row += [f"{normal[i][0]:.3f}", f"{normal[i][1]:.3f}"]
            else:
                row += ["", ""]
            w.writerow(row)


def main() -> None:
    buggy = resample_and_smooth(load_power_series(BUGGY_TRACE))
    normal = resample_and_smooth(load_power_series(NORMAL_TRACE))
    write_csv(OUTPUT_CSV, buggy, normal)
    print(f"buggy:  {len(buggy)} rows, last_ts={buggy[-1][0]:.0f} us")
    print(f"normal: {len(normal)} rows, last_ts={normal[-1][0]:.0f} us")
    print(f"wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
