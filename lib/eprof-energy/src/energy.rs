//! Sampling GPU board power.
//!
//! A background thread polls instantaneous board power from NVML at
//! `RECORD_FREQ` Hz and appends `(timestamp, device, power_mW)`. Polling is the
//! only way to get this: NVML reports a level, not an integral, so energy over
//! a window has to be reconstructed from a dense enough series of levels. The
//! frequency is chosen just under 1 kHz for that reason.
//!
//! Which clock stamps a sample is the one decision here that reaches outside
//! this file -- see [`ClockSource`]. A sample is only useful if it can be
//! placed on the same timeline as the kernels it is meant to explain.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use nvml_wrapper::Nvml;

/// Initialize NVML, tolerating systems that ship only the runtime library
/// (`libnvidia-ml.so.1`) without the unversioned `libnvidia-ml.so` dev symlink.
fn init_nvml() -> Result<Nvml, nvml_wrapper::error::NvmlError> {
    Nvml::init().or_else(|_| {
        Nvml::builder()
            .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
            .init()
    })
}

/// Sampling frequency in Hz. Just under 1 kHz: fast enough that a kernel of a
/// few hundred microseconds still falls inside a sampling interval, slow enough
/// that the polling thread does not compete with the workload for the driver.
pub const RECORD_FREQ: u32 = 998;

/// Which clock stamps a sample.
///
/// A power sample is only meaningful next to the kernels it overlaps, so it has
/// to carry the same clock they do. torch stamps events with
/// `c10::getApproximateTime()` -- the raw `rdtsc` counter on x86_64 -- and
/// calibrates the counter to nanoseconds once, when the run ends. A sampler
/// feeding a run must therefore emit that same raw counter and let the run
/// convert it; stamping wall-clock nanoseconds here would put the samples on a
/// timeline nothing else in the trace shares.
///
/// `UnixNanos` exists for reading the series on its own, outside a run, where
/// a raw counter means nothing to a human.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockSource {
    /// Unix-epoch nanoseconds. Readable on its own, comparable to nothing else.
    UnixNanos,
    /// The raw counter torch stamps its events with, on the same timeline.
    ApproxTsc,
}

/// One reading: what the board was drawing, and when.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnergySample {
    /// Timestamp in the profiler's selected `ClockSource` domain.
    pub time_ns: u64,
    /// Device index this sample belongs to.
    pub device_index: u32,
    /// Instantaneous board power in milliwatts (mW).
    pub power_mw: u32,
}

/// Error type for the energy profiler.
#[derive(Debug)]
pub enum EnergyError {
    Nvml(nvml_wrapper::error::NvmlError),
    InvalidDeviceId(u32),
    AlreadyStarted,
    NotStarted,
}

impl std::fmt::Display for EnergyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnergyError::Nvml(e) => write!(f, "NVML error: {e}"),
            EnergyError::InvalidDeviceId(id) => write!(f, "Invalid device id: {id}"),
            EnergyError::AlreadyStarted => write!(f, "Already started a worker thread!"),
            EnergyError::NotStarted => write!(f, "Energy Profiler has not been started!"),
        }
    }
}

impl std::error::Error for EnergyError {}

impl From<nvml_wrapper::error::NvmlError> for EnergyError {
    fn from(e: nvml_wrapper::error::NvmlError) -> Self {
        EnergyError::Nvml(e)
    }
}

/// Samples GPU board power on a background thread.
pub struct EnergyProfiler {
    nvml: Arc<Nvml>,
    /// Device indices to sample (validated at construction).
    device_indices: Vec<u32>,
    clock: ClockSource,
    signal: Arc<AtomicBool>,
    worker: Option<JoinHandle<Vec<EnergySample>>>,
    records: Vec<EnergySample>,
}

impl EnergyProfiler {
    /// Initialize NVML and select devices, sampling in the given clock domain.
    /// An empty `device_ids` selects every visible device.
    pub fn new(device_ids: &[u32], clock: ClockSource) -> Result<Self, EnergyError> {
        let nvml = init_nvml()?;
        let device_count = nvml.device_count()?;

        let device_indices: Vec<u32> = if device_ids.is_empty() {
            (0..device_count).collect()
        } else {
            for &id in device_ids {
                if id >= device_count {
                    return Err(EnergyError::InvalidDeviceId(id));
                }
            }
            device_ids.to_vec()
        };

        // Validate each handle resolves now, so failures surface at construction
        // rather than inside the worker thread.
        for &i in &device_indices {
            let _ = nvml.device_by_index(i)?;
        }

        Ok(Self {
            nvml: Arc::new(nvml),
            device_indices,
            clock,
            signal: Arc::new(AtomicBool::new(false)),
            worker: None,
            records: Vec::new(),
        })
    }

    /// Start the background sampling thread.
    pub fn start(&mut self) -> Result<(), EnergyError> {
        if self.worker.is_some() {
            return Err(EnergyError::AlreadyStarted);
        }
        self.signal.store(false, Ordering::SeqCst);

        let nvml = Arc::clone(&self.nvml);
        let indices = self.device_indices.clone();
        let signal = Arc::clone(&self.signal);
        let clock = self.clock;
        let period = Duration::from_secs_f64(1.0 / RECORD_FREQ as f64);

        self.worker = Some(std::thread::spawn(move || {
            // Resolve device handles once for the lifetime of the thread. They
            // borrow `nvml`, which the closure owns, so both live and drop
            // together within this scope (no self-referential struct needed).
            let devices: Vec<_> = indices
                .iter()
                .map(|&i| nvml.device_by_index(i).expect("device handle vanished"))
                .collect();

            let mut local = Vec::new();
            while !signal.load(Ordering::SeqCst) {
                let t = now_ts(clock);
                for (&idx, dev) in indices.iter().zip(&devices) {
                    // power_usage() returns milliwatts, like nvmlDeviceGetPowerUsage.
                    if let Ok(power_mw) = dev.power_usage() {
                        local.push(EnergySample {
                            time_ns: t,
                            device_index: idx,
                            power_mw,
                        });
                    }
                }
                std::thread::sleep(period);
            }
            local
        }));
        Ok(())
    }

    /// Stop sampling, join the worker, and collect its records.
    pub fn stop(&mut self) -> Result<(), EnergyError> {
        let worker = self.worker.take().ok_or(EnergyError::NotStarted)?;
        self.signal.store(true, Ordering::SeqCst);
        // If the worker panicked, surface it as NotStarted-equivalent rather
        // than unwinding through the caller.
        self.records = worker.join().unwrap_or_default();
        Ok(())
    }

    /// The samples collected so far, in the order they were taken.
    pub fn records(&self) -> &[EnergySample] {
        &self.records
    }

    /// The device indices being sampled.
    pub fn device_indices(&self) -> &[u32] {
        &self.device_indices
    }
}

fn now_ts(clock: ClockSource) -> u64 {
    match clock {
        ClockSource::UnixNanos => now_unix_ns(),
        ClockSource::ApproxTsc => now_rdtsc(),
    }
}

fn now_unix_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Raw timestamp counter, matching `c10::getApproximateTime()` on x86_64
/// (`__rdtsc`). On other architectures we fall back to Unix nanoseconds; the
/// approximate clock torch uses is also non-TSC there, so the two still agree
/// within a single platform -- which is all the calibration needs.
#[inline]
fn now_rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: _rdtsc has no preconditions; it reads the timestamp counter.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        now_unix_ns()
    }
}
