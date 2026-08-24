//! Charging GPU time and energy to the operators that caused them.

use std::collections::HashMap;

const UNATTRIBUTED: &str = "<unattributed>";

/// One GPU kernel's timing and ownership.
#[derive(Clone, Copy)]
pub struct Kernel {
    pub start_ns: i64,
    pub end_ns: i64,
    pub device: i32,
    pub correlation_id: i64,
}

/// One operator, summed over every time it ran.
#[derive(Clone, Debug, PartialEq)]
pub struct OpRecord {
    pub op_name: String,
    pub num_calls: i64,
    pub num_kernels: i64,
    pub cpu_time_ns: i64,
    pub gpu_time_ns: i64,
    pub gpu_energy_j: f64,
}

/// Piecewise-linear board-power curve (W) over time (ns), for one device.
pub struct PowerTimeline {
    ts: Vec<i64>,
    w: Vec<f64>,
}

impl PowerTimeline {
    /// `samples` is (timestamp_ns, watts), sorted by time.
    pub fn new(mut samples: Vec<(i64, f64)>) -> Self {
        samples.sort_by_key(|&(t, _)| t);
        let ts = samples.iter().map(|&(t, _)| t).collect();
        let w = samples.iter().map(|&(_, p)| p).collect();
        Self { ts, w }
    }

    pub fn usable(&self) -> bool {
        self.ts.len() >= 2
    }

    /// Linearly interpolated power (W) at `t`, clamped at the ends.
    pub fn power_at(&self, t: i64) -> f64 {
        if self.ts.is_empty() {
            return 0.0;
        }
        if t <= self.ts[0] {
            return self.w[0];
        }
        if t >= *self.ts.last().unwrap() {
            return *self.w.last().unwrap();
        }
        // partition_point: first index with ts > t; i = that - 1.
        let i = self.ts.partition_point(|&x| x <= t) - 1;
        let (t0, t1) = (self.ts[i], self.ts[i + 1]);
        let (w0, w1) = (self.w[i], self.w[i + 1]);
        if t1 == t0 {
            return w0;
        }
        let frac = (t - t0) as f64 / (t1 - t0) as f64;
        w0 + frac * (w1 - w0)
    }

    fn sample_points_in(&self, lo: i64, hi: i64) -> Vec<i64> {
        let start = self.ts.partition_point(|&x| x <= lo);
        let mut out = Vec::new();
        let mut i = start;
        while i < self.ts.len() && self.ts[i] < hi {
            out.push(self.ts[i]);
            i += 1;
        }
        out
    }
}

/// Energy (J) per kernel index, board power split by concurrency.
fn energy_concurrency_split(kernels: &[Kernel], power: &PowerTimeline) -> Vec<f64> {
    let mut energy = vec![0.0_f64; kernels.len()];
    if kernels.is_empty() || !power.usable() {
        return energy;
    }

    let mut starts: HashMap<i64, Vec<usize>> = HashMap::new();
    let mut ends: HashMap<i64, Vec<usize>> = HashMap::new();
    let mut breakpoints: Vec<i64> = Vec::new();
    let lo = kernels.iter().map(|k| k.start_ns).min().unwrap();
    let hi = kernels.iter().map(|k| k.end_ns).max().unwrap();
    for (idx, k) in kernels.iter().enumerate() {
        starts.entry(k.start_ns).or_default().push(idx);
        ends.entry(k.end_ns).or_default().push(idx);
        breakpoints.push(k.start_ns);
        breakpoints.push(k.end_ns);
    }
    breakpoints.extend(power.sample_points_in(lo, hi));
    breakpoints.sort_unstable();
    breakpoints.dedup();

    let mut active: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for seg in breakpoints.windows(2) {
        let (t0, t1) = (seg[0], seg[1]);
        // [start, end): at t0 remove kernels ending here, then add those starting.
        if let Some(v) = ends.get(&t0) {
            for &idx in v {
                active.remove(&idx);
            }
        }
        if let Some(v) = starts.get(&t0) {
            for &idx in v {
                active.insert(idx);
            }
        }
        if active.is_empty() || t1 <= t0 {
            continue;
        }
        let dt_s = (t1 - t0) as f64 / 1e9;
        let p_avg = (power.power_at(t0) + power.power_at(t1)) / 2.0;
        let share = p_avg * dt_s / active.len() as f64;
        for &idx in &active {
            energy[idx] += share;
        }
    }
    energy
}

/// Inputs mirror what attribution.py extracts from the profiler result.
pub struct AttributionInput {
    /// (op_name, cpu_duration_ns) for each aten-op invocation.
    pub cpu_ops: Vec<(String, i64)>,
    pub kernels: Vec<Kernel>,
    pub power: Vec<(i64, i32, i64)>,
    pub corr_to_op: HashMap<i64, String>,
    pub device_index: Option<i32>,
}

/// Attribute per-op latency + energy. Returns records sorted by GPU energy desc.
pub fn attribute(input: &AttributionInput) -> Vec<OpRecord> {
    // Power timelines per device (W).
    let mut by_dev: HashMap<i32, Vec<(i64, f64)>> = HashMap::new();
    for &(t, dev, mw) in &input.power {
        by_dev.entry(dev).or_default().push((t, mw as f64 / 1000.0));
    }
    let timelines: HashMap<i32, PowerTimeline> = by_dev
        .into_iter()
        .map(|(d, s)| (d, PowerTimeline::new(s)))
        .collect();

    // Kernels filtered by device.
    let kernels: Vec<Kernel> = input
        .kernels
        .iter()
        .copied()
        .filter(|k| input.device_index.map_or(true, |d| k.device == d))
        .filter(|k| k.end_ns > k.start_ns)
        .collect();

    // Energy per kernel, integrated per device with concurrency splitting.
    let mut kernel_energy = vec![0.0_f64; kernels.len()];
    let mut dev_groups: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, k) in kernels.iter().enumerate() {
        dev_groups.entry(k.device).or_default().push(i);
    }
    let empty = PowerTimeline::new(vec![]);
    for (dev, idxs) in &dev_groups {
        let sub: Vec<Kernel> = idxs.iter().map(|&i| kernels[i]).collect();
        let e = energy_concurrency_split(&sub, timelines.get(dev).unwrap_or(&empty));
        for (local, &global) in idxs.iter().enumerate() {
            kernel_energy[global] = e[local];
        }
    }

    // Aggregate. Use an insertion-ordered map so ties are stable like Python.
    let mut order: Vec<String> = Vec::new();
    let mut records: HashMap<String, OpRecord> = HashMap::new();
    let rec = |name: &str, records: &mut HashMap<String, OpRecord>, order: &mut Vec<String>| {
        if !records.contains_key(name) {
            order.push(name.to_string());
            records.insert(
                name.to_string(),
                OpRecord {
                    op_name: name.to_string(),
                    num_calls: 0,
                    num_kernels: 0,
                    cpu_time_ns: 0,
                    gpu_time_ns: 0,
                    gpu_energy_j: 0.0,
                },
            );
        }
    };

    for (name, dur) in &input.cpu_ops {
        rec(name, &mut records, &mut order);
        let r = records.get_mut(name).unwrap();
        r.num_calls += 1;
        r.cpu_time_ns += (*dur).max(0);
    }
    for (i, k) in kernels.iter().enumerate() {
        let name = input
            .corr_to_op
            .get(&k.correlation_id)
            .map(|s| s.as_str())
            .unwrap_or(UNATTRIBUTED);
        rec(name, &mut records, &mut order);
        let r = records.get_mut(name).unwrap();
        r.num_kernels += 1;
        r.gpu_time_ns += k.end_ns - k.start_ns;
        r.gpu_energy_j += kernel_energy[i];
    }

    let mut out: Vec<OpRecord> = order.into_iter().map(|n| records.remove(&n).unwrap()).collect();
    // Sort by energy desc; stable to preserve insertion order on ties.
    out.sort_by(|a, b| b.gpu_energy_j.partial_cmp(&a.gpu_energy_j).unwrap());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_interpolation() {
        let tl = PowerTimeline::new(vec![(0, 100.0), (10, 200.0)]);
        assert_eq!(tl.power_at(-5), 100.0);
        assert_eq!(tl.power_at(15), 200.0);
        assert!((tl.power_at(5) - 150.0).abs() < 1e-9);
    }

    #[test]
    fn concurrency_split_conserves_energy() {
        // Constant 100 W board. Two kernels overlapping [0,10) and [5,15).
        let power = PowerTimeline::new(vec![(0, 100.0), (20_000_000_000, 100.0)]);
        let kernels = vec![
            Kernel { start_ns: 0, end_ns: 10_000_000_000, device: 0, correlation_id: 1 },
            Kernel { start_ns: 5_000_000_000, end_ns: 15_000_000_000, device: 0, correlation_id: 2 },
        ];
        let e = energy_concurrency_split(&kernels, &power);
        // Busy window [0,15)s at 100W = 1500 J, fully distributed.
        let total: f64 = e.iter().sum();
        assert!((total - 1500.0).abs() < 1e-6, "total={total}");
    }
}
