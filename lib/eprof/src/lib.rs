//! The profiler as python sees it.

mod attribution;
mod driver;
mod keep;

use std::collections::HashMap;
use std::ffi::CStr;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::attribution::{AttributionInput, Kernel};
use eprof_storage::event::ExportNode;
use eprof_storage::materialize::{
    materialize, reassign_kineto_parents as rust_reassign, FullNode, MergeNode,
};

/// An event's tag, as the name python knows it by.
fn tag_name(tag: i32) -> &'static str {
    match tag {
        0 => "TorchOp",
        1 => "Power",
        2 => "Allocation",
        3 => "OutOfMemory",
        4 => "PyCall",
        5 => "PyCCall",
        6 => "Kineto",
        _ => "Unknown",
    }
}

#[pyclass(name = "_TreeNode")]
struct TreeNode {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    tag: String,
    #[pyo3(get)]
    start_time_ns: i64,
    #[pyo3(get)]
    duration_time_ns: i64,
    #[pyo3(get)]
    correlation_id: u64,
    #[pyo3(get)]
    children: Vec<Py<TreeNode>>,
}

// The one thing here that is genuinely C++: installing CPython's profiling
// hook, which needs the interpreter's own headers.
extern "C" {
    fn eprof_tracer_init();
}

/// What a run can be asked to collect.
#[pyclass(name = "_ActivityType", eq, eq_int, hash, frozen)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ActivityType {
    CPU = 0,
    XPU = 1,
    CUDA = 2,
    MTIA = 3,
    PrivateUse1 = 4,
}

fn node_name(n: &ExportNode) -> String {
    if n.name.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(n.name) }.to_string_lossy().into_owned()
    }
}

/// What a run collected. Owns the events and the kineto trace.
#[pyclass(name = "_ProfilerResult", unsendable)]
struct ProfilerResult {
    inner: eprof_kineto::ProfilerResult,
}

#[pymethods]
impl ProfilerResult {
    /// When the kineto trace began, on the trace's own clock.
    fn trace_start_ns(&self) -> i64 {
        self.inner.trace_start_ns as i64
    }

    /// Writes the chrome trace. Raises if there is no trace to write.
    fn save(&mut self, path: &str) -> PyResult<()> {
        if !self.inner.save(std::path::Path::new(path)) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("missing trace"));
        }
        Ok(())
    }

    /// The materialized event tree as nested nodes.
    fn experimental_event_tree(&mut self, py: Python<'_>) -> PyResult<Vec<Py<TreeNode>>> {
        let nodes = self.inner.events().export().to_vec();
        let nodes = &nodes[..];
        let mut built: Vec<Py<TreeNode>> = Vec::with_capacity(nodes.len());
        let mut roots: Vec<Py<TreeNode>> = Vec::new();
        for n in nodes {
            let obj = Py::new(
                py,
                TreeNode {
                    name: node_name(n),
                    tag: tag_name(n.tag).to_string(),
                    start_time_ns: n.start_ns,
                    duration_time_ns: n.dur_ns,
                    correlation_id: n.correlation_id,
                    children: Vec::new(),
                },
            )?;
            match usize::try_from(n.parent_id).ok().and_then(|p| built.get(p)) {
                Some(parent) => parent.borrow_mut(py).children.push(obj.clone_ref(py)),
                None => roots.push(obj.clone_ref(py)),
            }
            built.push(obj);
        }
        Ok(roots)
    }

    /// The same tree flat: one dict per node, in pre-order.
    fn export_raw_nodes<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty_bound(py);
        let nodes = self.inner.events().export().to_vec();
        for n in &nodes {
            let d = PyDict::new_bound(py);
            d.set_item("id", n.id)?;
            d.set_item("parent_id", n.parent_id)?;
            d.set_item("tag", n.tag)?;
            d.set_item("name", node_name(n))?;
            d.set_item("start_tid", n.start_tid)?;
            d.set_item("forward_tid", n.forward_tid)?;
            d.set_item("start_ns", n.start_ns)?;
            d.set_item("dur_ns", n.dur_ns)?;
            d.set_item("correlation_id", n.correlation_id)?;
            d.set_item("device", n.device)?;
            d.set_item("device_type", n.device_type)?;
            d.set_item("device_index", n.device_index)?;
            d.set_item("power_usage", n.power_usage)?;
            d.set_item("resource", n.resource)?;
            d.set_item("flow_id", n.flow_id)?;
            d.set_item("flow_type", n.flow_type)?;
            d.set_item("flow_start", n.flow_start)?;
            d.set_item("linked_correlation", n.linked_correlation)?;
            d.set_item("linked_id", n.linked_id)?;
            d.set_item("activity_type", n.activity_type)?;
            out.append(d)?;
        }
        Ok(out)
    }
}

/// GPU power sampling on its own, without a profiling run.
#[pyclass(name = "_EnergySampler", unsendable)]
struct EnergySampler {
    inner: std::cell::RefCell<Option<eprof_energy::EnergyProfiler>>,
    device_ids: Vec<u32>,
}

#[pymethods]
impl EnergySampler {
    #[new]
    #[pyo3(signature = (device_ids=None))]
    fn new(device_ids: Option<Vec<u32>>) -> PyResult<Self> {
        Ok(EnergySampler {
            inner: std::cell::RefCell::new(None),
            device_ids: device_ids.unwrap_or_default(),
        })
    }

    /// Begins sampling. Raises if NVML is unavailable or already started.
    fn start(&self) -> PyResult<()> {
        if self.inner.borrow().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("already sampling"));
        }
        let mut sampler =
            eprof_energy::EnergyProfiler::new(&self.device_ids, eprof_energy::ClockSource::UnixNanos)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("could not start NVML: {e:?}"))
                })?;
        sampler.start().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("could not start sampling: {e:?}"))
        })?;
        *self.inner.borrow_mut() = Some(sampler);
        Ok(())
    }

    fn stop(&self) -> PyResult<Vec<(u64, i32, u32)>> {
        let mut held = self.inner.borrow_mut();
        let mut sampler = held
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not sampling"))?;
        let _ = sampler.stop();
        let position: HashMap<u32, i32> = self
            .device_ids
            .iter()
            .enumerate()
            .map(|(i, &nvml)| (nvml, i as i32))
            .collect();
        Ok(sampler
            .records()
            .iter()
            .map(|s| {
                let device = position
                    .get(&s.device_index)
                    .copied()
                    .unwrap_or(s.device_index as i32);
                (s.time_ns, device, s.power_mw)
            })
            .collect())
    }
}

/// One profiling run, and the power sampler that runs alongside it.
#[pyclass(name = "_Profiler", unsendable)]
struct Profiler {
    inner: std::cell::RefCell<driver::Profiler>,
    /// The power sampler, once started; None when energy profiling is off.
    energy: std::cell::RefCell<Option<eprof_energy::EnergyProfiler>>,
    profile_energy: bool,
    device_ids: Vec<u32>,
}

#[pymethods]
impl Profiler {
    #[new]
    fn new(
        activities: &Bound<'_, PyAny>,
        record_shapes: bool,
        with_flops: bool,
        profile_memory: bool,
        with_stack: bool,
        with_modules: bool,
        profile_energy: bool,
        device_ids: Vec<i32>,
    ) -> PyResult<Self> {
        let mut acts: Vec<i32> = Vec::new();
        for item in activities.iter()? {
            let a: ActivityType = item?.extract()?;
            acts.push(a as i32);
        }
        Ok(Profiler {
            inner: std::cell::RefCell::new(driver::Profiler::new(
                acts,
                record_shapes,
                with_flops,
                profile_memory,
                with_stack,
                with_modules,
            )),
            energy: std::cell::RefCell::new(None),
            profile_energy,
            device_ids: device_ids.iter().map(|d| *d as u32).collect(),
        })
    }

    #[pyo3(signature = (_scopes=None))]
    fn start(&self, _scopes: Option<Bound<'_, PyAny>>) {
        // Energy sampling runs beside the profiler, not inside it: the sampler
        // is its own thread and knows nothing about the run.
        if self.profile_energy {
            let clock = eprof_energy::ClockSource::ApproxTsc;
            if let Ok(mut e) = eprof_energy::EnergyProfiler::new(&self.device_ids, clock) {
                if e.start().is_ok() {
                    *self.energy.borrow_mut() = Some(e);
                }
            }
        }
        self.inner.borrow_mut().start();
    }

    /// Ends the run and returns what it collected.
    fn stop(&self) -> PyResult<ProfilerResult> {
        // Drain the power samples into the still-running profiler: they are
        // pushed onto the run's queue, so this has to happen before the stop.
        if let Some(mut e) = self.energy.borrow_mut().take() {
            let _ = e.stop();
            let prof = self.inner.borrow();
            let position: std::collections::HashMap<u32, i32> = self
                .device_ids
                .iter()
                .enumerate()
                .map(|(i, &nvml)| (nvml, i as i32))
                .collect();
            for s in e.records() {
                let device = position
                    .get(&s.device_index)
                    .copied()
                    .unwrap_or(s.device_index as i32);
                prof.report_power(s.time_ns as i64, device, s.power_mw as i64);
            }
        }
        let inner = self.inner.borrow_mut().stop();
        inner.map(|inner| ProfilerResult { inner }).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("the profiler collected nothing")
        })
    }

    /// Turns an activity on or off mid-run.
    fn toggle_config(&self, enable: bool, activities: Vec<ActivityType>) {
        let acts: Vec<i32> = activities.iter().map(|a| *a as i32).collect();
        self.inner.borrow_mut().toggle(enable, &acts);
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        // Both halves free themselves; the sampler's thread is joined by its
        // own Drop, which is why it has to go before anything else runs.
        drop(self.energy.borrow_mut().take());
    }
}


/// Charge latency and energy to the operators that caused them.
#[pyfunction]
#[pyo3(signature = (cpu_ops, kernels, power, corr_to_op, device_index=None))]
fn attribute(
    cpu_ops: Vec<(String, i64)>,
    kernels: Vec<(i64, i64, i32, i64)>,
    power: Vec<(i64, i32, i64)>,
    corr_to_op: HashMap<i64, String>,
    device_index: Option<i32>,
) -> Vec<(String, i64, i64, i64, i64, f64)> {
    let input = AttributionInput {
        cpu_ops,
        kernels: kernels
            .into_iter()
            .map(|(s, e, d, c)| Kernel {
                start_ns: s,
                end_ns: e,
                device: d,
                correlation_id: c,
            })
            .collect(),
        power,
        corr_to_op,
        device_index,
    };
    crate::attribution::attribute(&input)
        .into_iter()
        .map(|r| {
            (
                r.op_name,
                r.num_calls,
                r.num_kernels,
                r.cpu_time_ns,
                r.gpu_time_ns,
                r.gpu_energy_j,
            )
        })
        .collect()
}

/// Reassign Kineto-node parents via the flow/linked merge (R3).
#[pyfunction]
fn reassign_kineto_parents(
    nodes: Vec<(i64, i64, i32, u32, u32, u32, i64)>,
) -> Vec<i64> {
    let merge: Vec<MergeNode> = nodes
        .into_iter()
        .map(|(id, orig_parent_id, tag, flow_id, flow_type, flow_start, linked_id)| {
            MergeNode {
                id,
                orig_parent_id,
                tag,
                flow_id,
                flow_type,
                flow_start,
                linked_id,
            }
        })
        .collect();
    rust_reassign(&merge)
}

/// Full materialization (flow/linked merge + build_tree containment).
#[pyfunction]
#[pyo3(name = "materialize")]
#[pyo3(signature = (nodes, current_tid=0))]
fn py_materialize(
    nodes: Vec<(i64, i32, u64, u64, i64, i64, u32, u32, u32, i64)>,
    current_tid: u64,
) -> Vec<i64> {
    let full: Vec<FullNode> = nodes
        .into_iter()
        .map(
            |(id, tag, start_tid, forward_tid, start_ns, end_ns, flow_id, flow_type, flow_start, linked_id)| {
                FullNode {
                    id,
                    tag,
                    start_tid,
                    forward_tid,
                    start_ns,
                    end_ns,
                    flow_id,
                    flow_type,
                    flow_start,
                    linked_id,
                }
            },
        )
        .collect();
    materialize(&full, current_tid).parents
}

#[pymodule]
fn magneton_eprof(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Registers the tracer's python type. Once, when the module loads.
    unsafe { eprof_tracer_init() };
    m.add_class::<ActivityType>()?;
    m.add_class::<Profiler>()?;
    m.add_class::<ProfilerResult>()?;
    m.add_class::<TreeNode>()?;
    m.add_class::<EnergySampler>()?;
    m.add_function(wrap_pyfunction!(attribute, m)?)?;
    m.add_function(wrap_pyfunction!(reassign_kineto_parents, m)?)?;
    m.add_function(wrap_pyfunction!(py_materialize, m)?)?;
    Ok(())
}
