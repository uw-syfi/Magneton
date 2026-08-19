//! Interning a callsite, so that describing one is paid for once.
//!
//! With tracing on, python calls a hook on every frame entry. Describing a
//! frame -- the file and line, the function name, the class of the module being
//! called -- means several CPython lookups and several string allocations, and
//! doing that per call would dominate the measurement of a program that makes
//! millions of them.
//!
//! But a callsite is described the same way every time it is reached. So the
//! tracer builds a cheap key out of identities it already has (interned string
//! pointers, `PyObject` addresses), looks it up here, and on a hit records
//! nothing but the resulting trace key. The lookups and the allocations happen
//! on a miss, once per distinct callsite for the whole run.
//!
//! Everything here runs under the GIL, because CPython profile hooks always
//! hold it. That is what lets one unsynchronised cache be shared by every
//! traced thread.
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};

/// A python code location, identified the way CPython lets us identify one
/// cheaply: by the *identity* of the interned filename and name strings, plus
/// the line. Comparing two pointers is what makes this usable on the hot path.
/// The strings behind them are stored once, on the first miss.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct CodeLoc {
    pub filename: u64,
    pub name: u64,
    pub line: i32,
}

/// Call kinds, in the order the tracer distinguishes them. A module call is
/// still a python call in the event stream; it just resolves its description
/// from `nn.Module.__call__` plus the module instance rather than from its own
/// frame.
pub const PY_CALL: u8 = 0;
pub const PY_MODULE_CALL: u8 = 1;
pub const PY_C_CALL: u8 = 2;

/// Event tags, mirroring EventType in event.rs.
pub(crate) const K_PY_CALL: i32 = 4;
pub(crate) const K_PY_C_CALL: i32 = 5;

struct FrameState {
    line: i32,
    filename: CString,
    funcname: CString,
}

/// What the tracer interns. `value_loc` is used by plain python calls, whose
/// identity *is* a code location; `value_ptr` by the two kinds keyed on a
/// PyObject identity instead (the module instance, the bound C function).
///
/// `python_tid` is part of the key, not just payload: two threads entering the
/// same callsite must get different trace keys, because the replay in
/// post-processing keeps one stack per python thread and would otherwise
/// interleave them onto one.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SiteKey {
    pub(crate) call_type: u8,
    pub(crate) python_tid: u64,
    value_loc: CodeLoc,
    value_ptr: u64,
    caller: CodeLoc,
}

/// A resolved callsite, as post-processing needs it. Strings point into the
/// cache and stay valid until it is destroyed.
#[repr(C)]
pub struct SiteView {
    pub key: u64,
    pub event_type: i32,
    pub python_tid: u64,
    pub caller_line: i32,
    pub caller_filename: *const c_char,
    pub caller_name: *const c_char,
    pub callsite_line: i32,
    pub callsite_filename: *const c_char,
    pub callsite_name: *const c_char,
    pub has_module: i32,
    pub module_cls_name: *const c_char,
    pub module_id: u64,
    pub function_name: *const c_char,
}

#[derive(Default)]
pub struct PyCache {
    locations: HashMap<CodeLoc, FrameState>,
    /// `nn.Module.__call__`'s own location, shared by every module call.
    module_location: Option<CodeLoc>,
    /// module instance -> its class.
    module_cls: HashMap<u64, u64>,
    cls_names: HashMap<u64, CString>,
    /// class -> (instance -> per-class instance number).
    module_ids: HashMap<u64, HashMap<u64, u64>>,
    /// bound C function -> its repr.
    c_names: HashMap<u64, CString>,
    keys: HashMap<SiteKey, u64>,
    /// Interned sites, in intern order. A trace key is an index into this plus
    /// one, so zero can mean "not interned". Read by the replay, which needs a
    /// site's call type and thread before it resolves the description.
    pub(crate) sites: Vec<SiteKey>,
    empty: CString,
}

impl PyCache {
    pub fn put_location(&mut self, loc: CodeLoc, filename: CString, funcname: CString) {
        self.locations.entry(loc).or_insert(FrameState {
            line: loc.line,
            filename,
            funcname,
        });
    }

    pub fn module_location(&self) -> Option<CodeLoc> {
        self.module_location
    }

    pub fn set_module_location(&mut self, loc: CodeLoc) {
        self.module_location = Some(loc);
    }

    pub fn put_module_class(&mut self, self_ptr: u64, cls: u64, cls_name: CString) {
        self.module_cls.insert(self_ptr, cls);
        self.cls_names.entry(cls).or_insert(cls_name);
    }

    pub fn put_c_name(&mut self, method: u64, name: CString) {
        self.c_names.entry(method).or_insert(name);
    }

    /// Zero when the site has not been interned yet, which is the tracer's
    /// signal to go do the CPython work and call `intern`.
    pub fn get(&self, key: &SiteKeyArgs) -> u64 {
        self.keys.get(&key.to_key()).copied().unwrap_or(0)
    }

    pub fn intern(&mut self, key: &SiteKeyArgs) -> u64 {
        let k = key.to_key();
        if let Some(&existing) = self.keys.get(&k) {
            return existing;
        }
        self.sites.push(k);
        let trace_key = self.sites.len() as u64;
        self.keys.insert(k, trace_key);
        trace_key
    }

    /// Strips the longest-listed-first source roots off every stored filename,
    /// so a stack reads as `torch/nn/modules/module.py` rather than the full
    /// install path. Must run before any site is resolved, since views hand
    /// out pointers to these strings.
    pub fn trim_prefixes(&mut self, prefixes: &[String]) {
        for state in self.locations.values_mut() {
            let Ok(name) = state.filename.to_str() else {
                continue;
            };
            if let Some(p) = prefixes.iter().find(|p| name.starts_with(p.as_str())) {
                let trimmed = name[p.len()..].to_string();
                state.filename = CString::new(trimmed).unwrap_or_default();
            }
        }
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Resolves site `i`. Takes `&mut self` because a module's instance number
    /// is handed out here, on first resolve, rather than while tracing -- the
    /// tracer would otherwise pay for numbering it may never use.
    pub fn site(&mut self, i: usize) -> Option<SiteView> {
        let site = *self.sites.get(i)?;
        let empty = self.empty.as_ptr();

        let (caller_line, caller_filename, caller_name) = match self.locations.get(&site.caller) {
            Some(f) => (f.line, f.filename.as_ptr(), f.funcname.as_ptr()),
            None => (0, empty, empty),
        };

        let mut view = SiteView {
            key: i as u64 + 1,
            event_type: K_PY_CALL,
            python_tid: site.python_tid,
            caller_line,
            caller_filename,
            caller_name,
            callsite_line: 0,
            callsite_filename: empty,
            callsite_name: empty,
            has_module: 0,
            module_cls_name: empty,
            module_id: 0,
            function_name: empty,
        };

        match site.call_type {
            PY_CALL => {
                if let Some(f) = self.locations.get(&site.value_loc) {
                    view.callsite_line = f.line;
                    view.callsite_filename = f.filename.as_ptr();
                    view.callsite_name = f.funcname.as_ptr();
                }
            }
            PY_MODULE_CALL => {
                if let Some(f) = self.module_location.and_then(|l| self.locations.get(&l)) {
                    view.callsite_line = f.line;
                    view.callsite_filename = f.filename.as_ptr();
                    view.callsite_name = f.funcname.as_ptr();
                }
                if let Some(&cls) = self.module_cls.get(&site.value_ptr) {
                    let ids = self.module_ids.entry(cls).or_default();
                    let next = ids.len() as u64;
                    view.module_id = *ids.entry(site.value_ptr).or_insert(next);
                    view.has_module = 1;
                    view.module_cls_name =
                        self.cls_names.get(&cls).map_or(empty, |n| n.as_ptr());
                }
            }
            _ => {
                view.event_type = K_PY_C_CALL;
                view.function_name = self.c_names.get(&site.value_ptr).map_or(empty, |n| n.as_ptr());
            }
        }
        Some(view)
    }
}

/// The intern/lookup key as it crosses the ABI.
#[repr(C)]
pub struct SiteKeyArgs {
    pub call_type: u8,
    pub python_tid: u64,
    pub value_loc: CodeLoc,
    pub value_ptr: u64,
    pub caller: CodeLoc,
}

impl SiteKeyArgs {
    fn to_key(&self) -> SiteKey {
        SiteKey {
            call_type: self.call_type,
            python_tid: self.python_tid,
            value_loc: self.value_loc,
            value_ptr: self.value_ptr,
            caller: self.caller,
        }
    }
}

/// Everything the cache might need to describe a callsite it has not seen.
/// The tracer fills this once, on a miss, and the cache uses only the parts it
/// is actually missing -- which is cheaper than asking first: a miss happens
/// once per callsite for the whole run, and each question was a boundary
/// crossing of its own.
#[repr(C)]
pub struct SiteRecord {
    pub key: SiteKeyArgs,
    pub caller_filename: *const c_char,
    pub caller_funcname: *const c_char,
    /// A plain call's own location, or `nn.Module.__call__`'s for a module call.
    pub value_filename: *const c_char,
    pub value_funcname: *const c_char,
    /// Module calls only: where `nn.Module.__call__` is, and the instance class.
    pub module_loc: CodeLoc,
    pub module_cls: u64,
    pub module_cls_name: *const c_char,
    /// C calls only.
    pub c_function_name: *const c_char,
}

// --- C ABI ------------------------------------------------------------------

unsafe fn cstr(p: *const c_char) -> CString {
    if p.is_null() {
        CString::default()
    } else {
        CStr::from_ptr(p).to_owned()
    }
}

/// # Safety
/// Release with `eprof_pycache_destroy`.
#[no_mangle]
pub extern "C" fn eprof_pycache_create() -> *mut PyCache {
    Box::into_raw(Box::new(PyCache::default()))
}

/// # Safety
/// `cache` must come from `eprof_pycache_create` and not be used after.
#[no_mangle]
pub unsafe extern "C" fn eprof_pycache_destroy(cache: *mut PyCache) {
    if !cache.is_null() {
        drop(Box::from_raw(cache));
    }
}

/// # Safety
/// `cache` and `key` must be valid.
#[no_mangle]
pub unsafe extern "C" fn eprof_pycache_get(
    cache: *const PyCache,
    key: *const SiteKeyArgs,
) -> u64 {
    if cache.is_null() || key.is_null() {
        return 0;
    }
    (*cache).get(&*key)
}

/// # Safety
/// `cache` must be valid; `prefixes` must point to `n` NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn eprof_pycache_trim_prefixes(
    cache: *mut PyCache,
    prefixes: *const *const c_char,
    n: usize,
) {
    if cache.is_null() || prefixes.is_null() {
        return;
    }
    let list: Vec<String> = (0..n)
        .filter_map(|i| {
            let p = *prefixes.add(i);
            if p.is_null() {
                None
            } else {
                CStr::from_ptr(p).to_str().ok().map(str::to_string)
            }
        })
        .collect();
    (*cache).trim_prefixes(&list);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(filename: u64, name: u64, line: i32) -> CodeLoc {
        CodeLoc {
            filename,
            name,
            line,
        }
    }

    fn args(call_type: u8, python_tid: u64, value_loc: CodeLoc, value_ptr: u64) -> SiteKeyArgs {
        SiteKeyArgs {
            call_type,
            python_tid,
            value_loc,
            value_ptr,
            caller: loc(9, 9, 1),
        }
    }

    fn cs(s: &str) -> CString {
        CString::new(s).unwrap()
    }

    unsafe fn read(p: *const c_char) -> String {
        CStr::from_ptr(p).to_str().unwrap().to_string()
    }

    #[test]
    fn a_repeated_callsite_interns_to_the_same_key() {
        let mut c = PyCache::default();
        let a = args(PY_CALL, 0, loc(1, 2, 3), 0);
        assert_eq!(c.get(&a), 0, "must miss before it is interned");
        let k = c.intern(&a);
        assert_ne!(k, 0);
        assert_eq!(c.get(&a), k);
        assert_eq!(c.intern(&a), k, "interning twice must not allocate a key");
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn the_same_callsite_on_two_threads_gets_two_keys() {
        // Post-processing replays one stack per python thread, so a shared key
        // would put two threads' frames on one stack.
        let mut c = PyCache::default();
        let k0 = c.intern(&args(PY_CALL, 0, loc(1, 2, 3), 0));
        let k1 = c.intern(&args(PY_CALL, 1, loc(1, 2, 3), 0));
        assert_ne!(k0, k1);
    }

    #[test]
    fn a_python_call_resolves_to_its_own_frame_and_its_caller() {
        let mut c = PyCache::default();
        c.put_location(loc(1, 2, 30), cs("/a/train.py"), cs("step"));
        c.put_location(loc(9, 9, 1), cs("/a/main.py"), cs("<module>"));
        c.intern(&args(PY_CALL, 0, loc(1, 2, 30), 0));
        let v = c.site(0).unwrap();
        unsafe {
            assert_eq!(v.event_type, K_PY_CALL);
            assert_eq!(read(v.callsite_filename), "/a/train.py");
            assert_eq!(read(v.callsite_name), "step");
            assert_eq!(v.callsite_line, 30);
            assert_eq!(read(v.caller_name), "<module>");
            assert_eq!(v.has_module, 0);
        }
    }

    #[test]
    fn a_module_call_reads_its_description_off_the_class_not_the_frame() {
        let mut c = PyCache::default();
        c.put_location(loc(5, 5, 1), cs("torch/nn/modules/module.py"), cs("_call_impl"));
        c.set_module_location(loc(5, 5, 1));
        c.put_location(loc(9, 9, 1), cs("/a/main.py"), cs("<module>"));
        c.put_module_class(0xAAAA, 0xC15, cs("Linear"));
        c.intern(&args(PY_MODULE_CALL, 0, CodeLoc::default(), 0xAAAA));
        let v = c.site(0).unwrap();
        unsafe {
            assert_eq!(v.event_type, K_PY_CALL, "a module call is still a py call");
            assert_eq!(read(v.callsite_name), "_call_impl");
            assert_eq!(v.has_module, 1);
            assert_eq!(read(v.module_cls_name), "Linear");
            assert_eq!(v.module_id, 0);
        }
    }

    #[test]
    fn instances_of_one_class_are_numbered_from_zero_in_resolve_order() {
        let mut c = PyCache::default();
        c.put_module_class(0xA, 0xC15, cs("Linear"));
        c.put_module_class(0xB, 0xC15, cs("Linear"));
        c.put_module_class(0xD, 0xC0D, cs("Conv2d"));
        c.intern(&args(PY_MODULE_CALL, 0, CodeLoc::default(), 0xB));
        c.intern(&args(PY_MODULE_CALL, 0, CodeLoc::default(), 0xA));
        c.intern(&args(PY_MODULE_CALL, 0, CodeLoc::default(), 0xD));
        assert_eq!(c.site(0).unwrap().module_id, 0);
        assert_eq!(c.site(1).unwrap().module_id, 1);
        assert_eq!(
            c.site(2).unwrap().module_id,
            0,
            "numbering is per class, not global"
        );
        assert_eq!(c.site(0).unwrap().module_id, 0, "and is stable");
    }

    #[test]
    fn a_c_call_resolves_to_its_repr_and_is_tagged_as_one() {
        let mut c = PyCache::default();
        c.put_c_name(0xF00, cs("<built-in method randn>"));
        c.intern(&args(PY_C_CALL, 0, CodeLoc::default(), 0xF00));
        let v = c.site(0).unwrap();
        unsafe {
            assert_eq!(v.event_type, K_PY_C_CALL);
            assert_eq!(read(v.function_name), "<built-in method randn>");
        }
    }

    #[test]
    fn trimming_strips_the_first_matching_prefix_only() {
        let mut c = PyCache::default();
        c.put_location(loc(1, 1, 1), cs("/usr/lib/python3/torch/nn.py"), cs("f"));
        c.put_location(loc(2, 2, 1), cs("/home/me/train.py"), cs("g"));
        c.trim_prefixes(&["/usr/lib/python3/".to_string()]);
        c.intern(&args(PY_CALL, 0, loc(1, 1, 1), 0));
        c.intern(&args(PY_CALL, 0, loc(2, 2, 1), 0));
        unsafe {
            assert_eq!(read(c.site(0).unwrap().callsite_filename), "torch/nn.py");
            assert_eq!(
                read(c.site(1).unwrap().callsite_filename),
                "/home/me/train.py",
                "a filename under no listed root is left alone"
            );
        }
    }

    #[test]
    fn an_unresolvable_site_yields_empty_strings_rather_than_null() {
        // C++ passes these straight to the event array, so they must be
        // readable even when the tracer never stored the location.
        let mut c = PyCache::default();
        c.intern(&args(PY_CALL, 0, loc(1, 2, 3), 0));
        let v = c.site(0).unwrap();
        unsafe {
            assert_eq!(read(v.callsite_filename), "");
            assert_eq!(read(v.caller_filename), "");
        }
    }

    #[test]
    fn resolving_past_the_end_is_none() {
        let mut c = PyCache::default();
        assert!(c.site(0).is_none());
    }
}

/// Interns a callsite the cache has not seen, storing whatever of `rec` it is
/// missing.
///
/// Everything about the callsite arrives in one struct so that a miss costs one
/// crossing. Asking what the cache already holds and then telling it the rest
/// would take up to seven -- a probe and a store for the location, the module
/// class and the C function name, then the intern itself -- and the caller is
/// holding the GIL throughout.
///
/// # Safety
/// `cache` must come from `eprof_pycache_create`; every string in `rec` must be
/// NUL-terminated or null.
#[no_mangle]
pub unsafe extern "C" fn eprof_pycache_intern_site(
    cache: *mut PyCache,
    rec: *const SiteRecord,
) -> u64 {
    if cache.is_null() || rec.is_null() {
        return 0;
    }
    let cache = &mut *cache;
    let rec = &*rec;

    cache.put_location(
        rec.key.caller,
        cstr(rec.caller_filename),
        cstr(rec.caller_funcname),
    );

    match rec.key.call_type {
        PY_CALL => cache.put_location(
            rec.key.value_loc,
            cstr(rec.value_filename),
            cstr(rec.value_funcname),
        ),
        PY_MODULE_CALL => {
            if cache.module_location().is_none() {
                cache.put_location(
                    rec.module_loc,
                    cstr(rec.value_filename),
                    cstr(rec.value_funcname),
                );
                cache.set_module_location(rec.module_loc);
            }
            cache.put_module_class(
                rec.key.value_ptr,
                rec.module_cls,
                cstr(rec.module_cls_name),
            );
        }
        _ => cache.put_c_name(rec.key.value_ptr, cstr(rec.c_function_name)),
    }
    cache.intern(&rec.key)
}
