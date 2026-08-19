//! libkineto's activity types, and what device each one belongs to.

// --- libkineto::ActivityType (libkineto/include/ActivityType.h) --------------

pub mod act {
    pub const CPU_OP: i32 = 0;
    pub const USER_ANNOTATION: i32 = 1;
    pub const GPU_USER_ANNOTATION: i32 = 2;
    pub const GPU_MEMCPY: i32 = 3;
    pub const GPU_MEMSET: i32 = 4;
    pub const CONCURRENT_KERNEL: i32 = 5;
    pub const EXTERNAL_CORRELATION: i32 = 6;
    pub const CUDA_RUNTIME: i32 = 7;
    pub const CUDA_DRIVER: i32 = 8;
    pub const CPU_INSTANT_EVENT: i32 = 9;
    pub const PYTHON_FUNCTION: i32 = 10;
    pub const OVERHEAD: i32 = 11;
    pub const MTIA_RUNTIME: i32 = 12;
    pub const MTIA_CCP_EVENTS: i32 = 13;
    pub const CUDA_SYNC: i32 = 14;
    pub const GLOW_RUNTIME: i32 = 15;
    pub const CUDA_PROFILER_RANGE: i32 = 16;
    pub const XPU_RUNTIME: i32 = 18;
    pub const COLLECTIVE_COMM: i32 = 19;
    pub const MTIA_WORKLOADD: i32 = 20;
    pub const PRIVATEUSE1_RUNTIME: i32 = 21;
    pub const PRIVATEUSE1_DRIVER: i32 = 22;
}

/// What CPU profiling asks libkineto to record.
pub(crate) const CPU_TYPES: &[i32] = &[
    act::CPU_OP,
    act::CPU_INSTANT_EVENT,
    act::USER_ANNOTATION,
    act::EXTERNAL_CORRELATION,
    act::XPU_RUNTIME,
    act::CUDA_RUNTIME,
    act::CUDA_DRIVER,
    act::PYTHON_FUNCTION,
    act::PRIVATEUSE1_RUNTIME,
    act::PRIVATEUSE1_DRIVER,
];

/// What GPU profiling adds. CUDA_RUNTIME and CUDA_DRIVER appear in both: a
/// runtime call is a CPU-side event that is only interesting when there are
/// kernels to correlate it with.
pub(crate) const CUDA_TYPES: &[i32] = &[
    act::GPU_MEMCPY,
    act::GPU_MEMSET,
    act::GPU_USER_ANNOTATION,
    act::CONCURRENT_KERNEL,
    act::CUDA_RUNTIME,
    act::CUDA_DRIVER,
    act::OVERHEAD,
];

// --- c10::DeviceType, the values the event payload carries ------------------

const DEVICE_CPU: i8 = 0;
const DEVICE_CUDA: i8 = 1;
const DEVICE_MTIA: i8 = 19;
const DEVICE_PRIVATEUSE1: i8 = 20;

/// The device an activity belongs to. `privateuse1` says whether a custom
/// backend has claimed the PrivateUse1 slot -- those backends reuse the CUDA
/// and MTIA activity types, so the type alone does not identify the device.
pub fn device_type_from_activity(activity_type: i32, privateuse1: bool) -> i8 {
    match activity_type {
        act::GPU_MEMCPY
        | act::GPU_MEMSET
        | act::CONCURRENT_KERNEL
        | act::CUDA_SYNC
        | act::GPU_USER_ANNOTATION
        | act::CUDA_PROFILER_RANGE => {
            if privateuse1 {
                DEVICE_PRIVATEUSE1
            } else {
                DEVICE_CUDA
            }
        }
        act::MTIA_CCP_EVENTS | act::MTIA_WORKLOADD => {
            if privateuse1 {
                DEVICE_PRIVATEUSE1
            } else {
                DEVICE_MTIA
            }
        }
        act::CPU_OP
        | act::USER_ANNOTATION
        | act::EXTERNAL_CORRELATION
        | act::CUDA_RUNTIME
        | act::XPU_RUNTIME
        | act::CPU_INSTANT_EVENT
        | act::GLOW_RUNTIME
        | act::MTIA_RUNTIME
        | act::PYTHON_FUNCTION
        | act::CUDA_DRIVER
        | act::PRIVATEUSE1_RUNTIME
        | act::PRIVATEUSE1_DRIVER
        | act::OVERHEAD => DEVICE_CPU,
        other => {
            eprintln!("[eprof] unknown activity type ({other}), assuming CPU device");
            DEVICE_CPU
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kernel_belongs_to_the_gpu_and_a_runtime_call_does_not() {
        // The runtime call is the CPU-side launch; only the kernel runs on the
        // device, and that difference is what keeps launches in the CPU tree.
        assert_eq!(device_type_from_activity(act::CONCURRENT_KERNEL, false), DEVICE_CUDA);
        assert_eq!(device_type_from_activity(act::GPU_MEMCPY, false), DEVICE_CUDA);
        assert_eq!(device_type_from_activity(act::CUDA_RUNTIME, false), DEVICE_CPU);
        assert_eq!(device_type_from_activity(act::CPU_OP, false), DEVICE_CPU);
        assert_eq!(device_type_from_activity(act::PYTHON_FUNCTION, false), DEVICE_CPU);
    }

    #[test]
    fn a_privateuse1_backend_claims_the_reused_activity_types() {
        assert_eq!(
            device_type_from_activity(act::CONCURRENT_KERNEL, true),
            DEVICE_PRIVATEUSE1
        );
        assert_eq!(
            device_type_from_activity(act::MTIA_WORKLOADD, true),
            DEVICE_PRIVATEUSE1
        );
        assert_eq!(device_type_from_activity(act::MTIA_WORKLOADD, false), DEVICE_MTIA);
        assert_eq!(
            device_type_from_activity(act::CPU_OP, true),
            DEVICE_CPU,
            "a CPU type is never reused by a device backend"
        );
    }

    #[test]
    fn an_unknown_activity_type_falls_back_to_cpu() {
        assert_eq!(device_type_from_activity(9999, false), DEVICE_CPU);
    }
}
