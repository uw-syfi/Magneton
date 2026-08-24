//! Reads the sampler on its own, without a profiling run.

use std::collections::BTreeMap;
use std::env;
use std::thread::sleep;
use std::time::Duration;

use eprof_energy::{ClockSource, EnergyProfiler, EnergySample, RECORD_FREQ};

fn main() {
    let mut args = env::args().skip(1);
    let duration_ms: u64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let device_ids: Vec<u32> = args.filter_map(|s| s.parse().ok()).collect();

    println!(
        "eprof probe: sampling for {duration_ms} ms at {RECORD_FREQ} Hz \
         (devices: {})",
        if device_ids.is_empty() {
            "all".to_string()
        } else {
            format!("{device_ids:?}")
        }
    );

    let mut profiler = match EnergyProfiler::new(&device_ids, ClockSource::UnixNanos) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to init EnergyProfiler: {e}");
            std::process::exit(1);
        }
    };
    println!("monitoring devices: {:?}", profiler.device_indices());

    if let Err(e) = profiler.start() {
        eprintln!("failed to start: {e}");
        std::process::exit(1);
    }
    sleep(Duration::from_millis(duration_ms));
    if let Err(e) = profiler.stop() {
        eprintln!("failed to stop: {e}");
        std::process::exit(1);
    }

    let records = profiler.records();
    if records.is_empty() {
        eprintln!("no samples collected");
        std::process::exit(1);
    }

    // Group by device.
    let mut by_device: BTreeMap<u32, Vec<&EnergySample>> = BTreeMap::new();
    for s in records {
        by_device.entry(s.device_index).or_default().push(s);
    }

    let n_devices = by_device.len() as u64;
    let per_device_samples = records.len() as u64 / n_devices.max(1);
    let achieved_hz = per_device_samples as f64 / (duration_ms as f64 / 1000.0);

    println!(
        "\ncollected {} samples across {} device(s) \
         (~{} per device, ~{:.0} Hz/device vs {} Hz target)",
        records.len(),
        n_devices,
        per_device_samples,
        achieved_hz,
        RECORD_FREQ
    );

    println!("\n{:<8} {:>10} {:>10} {:>10} {:>12}", "device", "samples", "min_W", "max_W", "avg_W");
    println!("{}", "-".repeat(54));
    for (dev, samples) in &by_device {
        let powers: Vec<f64> = samples.iter().map(|s| s.power_mw as f64 / 1000.0).collect();
        let min = powers.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = powers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg = powers.iter().sum::<f64>() / powers.len() as f64;
        println!(
            "{:<8} {:>10} {:>10.2} {:>10.2} {:>12.2}",
            dev,
            samples.len(),
            min,
            max,
            avg
        );
    }

    // Rough energy estimate per device via trapezoidal integration over the
    // sampled window (sanity check; not the final attribution path).
    println!("\nwindow energy estimate (trapezoidal integration of board power):");
    for (dev, samples) in &by_device {
        let energy_j = trapezoid_energy_j(samples);
        println!("  device {dev}: {energy_j:.3} J");
    }
}

/// Integrate power (W) over time (s) using the trapezoid rule -> energy (J).
fn trapezoid_energy_j(samples: &[&EnergySample]) -> f64 {
    let mut energy = 0.0;
    for w in samples.windows(2) {
        let dt_s = (w[1].time_ns.saturating_sub(w[0].time_ns)) as f64 / 1e9;
        let p_avg_w = (w[0].power_mw as f64 + w[1].power_mw as f64) / 2.0 / 1000.0;
        energy += p_avg_w * dt_s;
    }
    energy
}
