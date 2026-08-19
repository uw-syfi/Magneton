//! A profiling run, from configuration to result.
//!
//! Starting and stopping a run means bringing up several independent things in
//! an order that matters: libkineto's trace has to be prepared before anything
//! is collected, the RecordFunction callbacks have to be attached after the
//! run's state exists, and the device has to be synchronized before the trace
//! ends or kernels still in flight are lost. This is where that order is
//! written down.
//!
//! Configuration is decided once, here, and never consulted again by the
//! collecting code -- the hot path is not the place to ask what was requested.

use eprof_kineto::{self as kineto, ProfilerResult};

/// The activity kinds a run can ask for. These numbers reach python as
/// `_ActivityType` and torch's profiler passes them through, so a value may be
/// added but never renumbered.
pub const ACTIVITY_CPU: i32 = 0;
pub const ACTIVITY_CUDA: i32 = 2;

fn activity_name(activity: i32) -> &'static str {
    match activity {
        0 => "CPU",
        1 => "XPU",
        2 => "CUDA",
        3 => "MTIA",
        4 => "PrivateUse1",
        _ => "?",
    }
}

pub struct Profiler {
    activities: Vec<i32>,
    /// The run in progress, or None between start and stop.
    run: Option<eprof_torch::run::Run>,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    entered: bool,
    trace_id: String,
}

impl Profiler {
    /// Configures a run and prepares kineto's trace. Panics if no activity was
    /// asked for -- there would be nothing to collect.
    pub fn new(
        activities: Vec<i32>,
        record_shapes: bool,
        with_flops: bool,
        profile_memory: bool,
        with_stack: bool,
        _with_modules: bool,
    ) -> Profiler {
        let mut list = activities;
        list.sort_unstable();
        list.dedup();
        assert!(!list.is_empty(), "No activities specified for the profiler");

        // with_flops and with_modules are accepted for compatibility with
        // torch.profiler's signature, but neither selects behaviour of its own.
        // Asking for flops implies input shapes, so it is folded into
        // record_shapes; module names come from the python tracer whenever it
        // runs, so with_modules has nothing left to turn on.
        let profiler = Profiler {
            record_shapes: record_shapes || with_flops,
            profile_memory,
            with_stack,
            entered: false,
            trace_id: uuid(),
            run: None,
            activities: list,
        };

        kineto::prepare_trace(
            /*cpu_only=*/ !profiler.wants(ACTIVITY_CUDA),
            profiler.wants(ACTIVITY_CPU),
            profiler.wants(ACTIVITY_CUDA),
            &profiler.trace_id,
        );
        profiler
    }

    fn wants(&self, activity: i32) -> bool {
        self.activities.contains(&activity)
    }

    pub fn start(&mut self) {
        assert!(!self.entered, "Profiler is already started");
        self.entered = true;

        // The python tracer is only worth running when python frames would be
        // recorded at all, which needs both the stack option and CPU activity.
        let has_cpu = self.wants(ACTIVITY_CPU);
        eprof_torch::run::set_record_shapes(self.record_shapes);
        // SAFETY: no run is in progress -- `entered` was just checked -- so
        // starting one and registering the callbacks is what the state allows.
        unsafe {
            self.run = Some(eprof_torch::run::Run::start(
                self.record_shapes,
                self.profile_memory,
                self.with_stack && has_cpu,
            ));
            if has_cpu {
                eprof_torch::run::callbacks_push();
            }
        }
        kineto::start_trace();
    }

    /// Ends the run and returns what it collected. None only if there was no
    /// run, which cannot happen after a successful `start`.
    pub fn stop(&mut self) -> Option<ProfilerResult> {
        assert!(self.entered, "Profiler is not started");
        if self.wants(ACTIVITY_CUDA) {
            // Kernels queued but not finished would otherwise be missing from
            // the trace, or land outside it.
            // SAFETY: no preconditions; it is a cudaDeviceSynchronize.
            unsafe { eprof_torch::run::device_synchronize() };
        }
        self.entered = false;

        let mut run = self.run.take()?;
        // SAFETY: the run is ours, still whole, and finished exactly once. It
        // hands over both the kineto trace and the event array it leaked at
        // start, and is dropped here without freeing either.
        unsafe {
            let trace = run.finish();
            Some(ProfilerResult::new(
                run.start_time_ns as u64,
                trace,
                Box::from_raw(run.events),
            ))
        }
    }

    /// Turns an activity on or off mid-run.
    pub fn toggle(&mut self, enable: bool, activities: &[i32]) {
        assert!(self.entered, "Profiler is not started");
        if activities.is_empty() {
            return;
        }
        let enable = i32::from(enable);
        let mut touched = 0;
        for &activity in activities {
            match activity {
                ACTIVITY_CUDA => {
                    touched += 1;
                    kineto::toggle(enable);
                }
                ACTIVITY_CPU => {
                    touched += 1;
                    if eprof_torch::run::state_active() {
                        // SAFETY: guarded on there being a state to attach to.
                        unsafe {
                            if enable != 0 {
                                eprof_torch::run::callbacks_push();
                            } else {
                                eprof_torch::run::callbacks_remove();
                            }
                        }
                    }
                }
                other => eprintln!(
                    "[eprof] Dynamic toggle is only supported for CPU/GPU activity, \
                     skipping toggling of {}",
                    activity_name(other)
                ),
            }
        }
        if touched == 1 {
            eprintln!(
                "[eprof] Toggling one of CPU/CUDA activity may result in incorrect \
                 correlation between CPU and CUDA events"
            );
        }
    }

    /// Records a power sample against the running run.
    ///
    /// Called while draining the sampler, before the run is stopped: the sample
    /// goes onto the collecting thread's subqueue like anything else, so it is
    /// sorted onto the same timeline as the ops and kernels it has to be
    /// compared against. Draining after the stop would have nowhere to put it.
    pub fn report_power(&self, time_ns: i64, device_index: i32, power_usage_mw: i64) {
        // c10::DeviceType::CUDA. The sampler only reads NVML.
        const DEVICE_CUDA: i8 = 1;
        let Some(run) = self.run.as_ref() else { return };
        // SAFETY: the run owns the queue and is alive for this call.
        unsafe {
            let sq = eprof_torch::queue::subqueue_for_current_thread(run.queue);
            (*sq).pods.push_power(eprof_torch::pods::PodPower {
                start_time: time_ns,
                power_usage_mw,
                device_type: DEVICE_CUDA,
                device_index: device_index as i8,
            });
        }
    }
}

/// A version-4 UUID, as the 32 uppercase hex digits kineto wants for a trace id.
fn uuid() -> String {
    // Two 64-bit draws, with the version and variant bits forced per RFC 4122.
    let mut hi = rand_u64();
    let mut lo = rand_u64();
    hi = (hi & 0xFFFF_FFFF_FFFF_0FFF) | 0x0000_0000_0000_4000;
    lo = (lo & 0x3FFF_FFFF_FFFF_FFFF) | 0x8000_0000_0000_0000;
    format!("{hi:016X}{lo:016X}")
}

/// xorshift64*, seeded from the address of a fresh allocation and a counter.
/// A trace id only has to be unlikely to collide, so this avoids pulling in an
/// rng crate for it.
fn rand_u64() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let seed = Box::into_raw(Box::new(0u8)) as u64;
    // Reclaim it -- we only wanted the address.
    unsafe { drop(Box::from_raw(seed as *mut u8)) };
    let mut x = seed ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2545_F491_4F6C_DD1D;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profiler(activities: Vec<i32>) -> Profiler {
        Profiler {
            activities,
            run: None,
            record_shapes: false,
            profile_memory: false,
            with_stack: false,
            entered: false,
            trace_id: String::new(),
        }
    }

    #[test]
    fn cuda_only_still_counts_as_wanting_cuda() {
        let p = profiler(vec![ACTIVITY_CUDA]);
        assert!(p.wants(ACTIVITY_CUDA));
        assert!(!p.wants(ACTIVITY_CPU));
    }

    #[test]
    fn an_unsupported_activity_is_named_in_the_toggle_warning() {
        assert_eq!(activity_name(3), "MTIA");
        assert_eq!(activity_name(99), "?");
    }
}
