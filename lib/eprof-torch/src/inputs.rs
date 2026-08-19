//! An op's recorded inputs, and the metadata strings they render to.
//!
//! Capture writes four parallel streams here, one group per op, terminated by
//! `TAG_TERMINATOR`: a tag per input, plus the payload for the tags that carry
//! one. Reading them back and rendering the "Input Dims" / "Input Strides" /
//! "Input type" / "Concrete Inputs" fields happens at materialization, which is
//! why the streams exist at all -- building those vectors per op while
//! profiling would distort what is being measured.
//!
//! Only the reading needs C++: deciding an input's tag, pulling a tensor's
//! dtype and sizes, printing a scalar. Those need an `at::Tensor` or a
//! `c10::IValue` in hand. Everything after -- storing, grouping, rendering --
//! is here.

use std::ffi::{c_char, CStr};

/// What an input is, which decides which of the streams below carries its
/// payload. Capture chooses the tag -- that is the part needing a `c10::IValue`
/// -- and everything after it reads these.
pub const TAG_TENSOR: u8 = 0;
pub const TAG_UNDEFINED_TENSOR: u8 = 1;
pub const TAG_TENSOR_LIST_BEGIN: u8 = 2;
pub const TAG_SCALAR_LIST: u8 = 3;
pub const TAG_SCALAR: u8 = 4;
pub const TAG_OTHER: u8 = 5;
pub const TAG_TERMINATOR: u8 = 6;

struct TensorMeta {
    dtype: String,
    /// Only a strided tensor has strides recorded after its sizes.
    strided: bool,
    ndim: usize,
}

/// A shape field. `Missing` is what an input with no shape of its own renders
/// as -- an empty `[]`, matching the default-constructed variant the C++ this
/// replaces produced for the same case.
enum Shape {
    Missing,
    One(Vec<i64>),
    Many(Vec<Vec<i64>>),
}

struct Input {
    shape: Shape,
    stride: Shape,
    dtype: String,
    concrete: String,
}

#[derive(Default)]
pub struct OpInputs {
    tags: Vec<u8>,
    tensors: Vec<TensorMeta>,
    /// Sizes then strides, `ndim` of each, per tensor in tag order.
    dims: Vec<i64>,
    /// Already-printed scalar values, in tag order.
    scalars: Vec<String>,

    // Read cursors. Materialization walks the streams once, one op per call,
    // so each op resumes where the last stopped.
    tag_at: usize,
    tensor_at: usize,
    dim_at: usize,
    scalar_at: usize,
}

impl OpInputs {
    pub fn push_tag(&mut self, tag: u8) {
        self.tags.push(tag);
    }

    pub fn push_tensor(&mut self, dtype: String, strided: bool, sizes: &[i64], strides: &[i64]) {
        self.tags.push(TAG_TENSOR);
        self.tensors.push(TensorMeta {
            dtype,
            strided,
            ndim: sizes.len(),
        });
        self.dims.extend_from_slice(sizes);
        if strided {
            self.dims.extend_from_slice(strides);
        }
    }

    pub fn push_scalar(&mut self, tag: u8, rendered: String) {
        self.tags.push(tag);
        self.scalars.push(rendered);
    }

    pub fn clear(&mut self) {
        self.tags.clear();
        self.tensors.clear();
        self.dims.clear();
        self.scalars.clear();
        self.tag_at = 0;
        self.tensor_at = 0;
        self.dim_at = 0;
        self.scalar_at = 0;
    }

    /// Consumes one tensor's dims. Returns None once the streams are out of
    /// step, which is reported rather than guessed at.
    fn take_tensor(&mut self) -> Option<(Vec<i64>, Vec<i64>, String)> {
        let meta = self.tensors.get(self.tensor_at)?;
        let (ndim, strided, dtype) = (meta.ndim, meta.strided, meta.dtype.clone());
        self.tensor_at += 1;
        let want = if strided { ndim * 2 } else { ndim };
        let block = self.dims.get(self.dim_at..self.dim_at + want)?;
        self.dim_at += want;
        let sizes = block[..ndim].to_vec();
        let strides = if strided { block[ndim..].to_vec() } else { Vec::new() };
        Some((sizes, strides, dtype))
    }

    /// Reads the next op's inputs off the streams. Returns an empty vec for an
    /// op with no inputs, which is what suppresses its metadata entirely.
    fn next_op(&mut self) -> Vec<Input> {
        let mut out = Vec::new();
        while let Some(&tag) = self.tags.get(self.tag_at) {
            self.tag_at += 1;
            match tag {
                TAG_TERMINATOR => break,
                TAG_TENSOR => {
                    let Some((sizes, strides, dtype)) = self.take_tensor() else {
                        break;
                    };
                    out.push(Input {
                        shape: Shape::One(sizes),
                        stride: Shape::One(strides),
                        dtype,
                        concrete: String::new(),
                    });
                }
                TAG_TENSOR_LIST_BEGIN => {
                    // The list's members follow, up to their own terminator.
                    let mut shapes = Vec::new();
                    let mut strides = Vec::new();
                    let mut undefined = false;
                    while let Some(&inner) = self.tags.get(self.tag_at) {
                        self.tag_at += 1;
                        match inner {
                            TAG_TERMINATOR => break,
                            TAG_UNDEFINED_TENSOR => undefined = true,
                            _ => {
                                let Some((s, st, _)) = self.take_tensor() else {
                                    break;
                                };
                                shapes.push(s);
                                strides.push(st);
                            }
                        }
                    }
                    // One undefined member and the whole list has no shape --
                    // a partial list would misrepresent the argument.
                    out.push(if undefined {
                        Input {
                            shape: Shape::Missing,
                            stride: Shape::Missing,
                            dtype: String::new(),
                            concrete: String::new(),
                        }
                    } else {
                        Input {
                            shape: Shape::Many(shapes),
                            stride: Shape::Many(strides),
                            dtype: "TensorList".into(),
                            concrete: String::new(),
                        }
                    });
                }
                TAG_SCALAR | TAG_SCALAR_LIST => {
                    let concrete = self
                        .scalars
                        .get(self.scalar_at)
                        .cloned()
                        .unwrap_or_default();
                    self.scalar_at += 1;
                    out.push(Input {
                        shape: Shape::Missing,
                        stride: Shape::Missing,
                        dtype: if tag == TAG_SCALAR { "Scalar" } else { "ScalarList" }.into(),
                        concrete,
                    });
                }
                // An undefined tensor, or anything we do not describe.
                _ => out.push(Input {
                    shape: Shape::Missing,
                    stride: Shape::Missing,
                    dtype: String::new(),
                    concrete: String::new(),
                }),
            }
        }
        out
    }

    /// The metadata an op's inputs render to, in the order they are written.
    /// Empty when the op recorded no inputs.
    pub fn next_metadata(&mut self) -> Vec<(String, String)> {
        let inputs = self.next_op();
        if inputs.is_empty() {
            return Vec::new();
        }
        vec![
            (
                "Input Dims".into(),
                render_shapes(inputs.iter().map(|i| &i.shape)),
            ),
            (
                "Input Strides".into(),
                render_shapes(inputs.iter().map(|i| &i.stride)),
            ),
            (
                "Input type".into(),
                render_str_list(inputs.iter().map(|i| i.dtype.as_str())),
            ),
            (
                "Concrete Inputs".into(),
                render_str_list(inputs.iter().map(|i| i.concrete.as_str())),
            ),
        ]
    }
}

fn write_dims(out: &mut String, dims: &[i64]) {
    out.push('[');
    for (i, d) in dims.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&d.to_string());
    }
    out.push(']');
}

/// A tensor list longer than this renders as `[]` rather than being spelled
/// out; matches TENSOR_LIST_DISPLAY_LENGTH_LIMIT below.
const TENSOR_LIST_DISPLAY_LENGTH_LIMIT: usize = 30;

fn render_shapes<'a>(shapes: impl Iterator<Item = &'a Shape>) -> String {
    let mut out = String::from("[");
    for (i, shape) in shapes.enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        match shape {
            // An input with no shape still occupies a slot, as an empty list.
            Shape::Missing => out.push_str("[]"),
            Shape::One(dims) => write_dims(&mut out, dims),
            Shape::Many(list) if list.len() > TENSOR_LIST_DISPLAY_LENGTH_LIMIT => {
                out.push_str("[]");
            }
            Shape::Many(list) => {
                out.push('[');
                for (k, dims) in list.iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    write_dims(&mut out, dims);
                }
                out.push(']');
            }
        }
    }
    out.push(']');
    out
}

fn render_str_list<'a>(items: impl Iterator<Item = &'a str>) -> String {
    let mut out = String::from("[");
    for (i, s) in items.enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('"');
        out.push_str(s);
        out.push('"');
    }
    out.push(']');
    out
}

// --- C ABI ------------------------------------------------------------------

unsafe fn read(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// # Safety
/// `h` must come from `eprof_op_inputs_create`.
#[no_mangle]
pub unsafe extern "C" fn eprof_op_inputs_push_tag(h: *mut OpInputs, tag: u8) {
    if !h.is_null() {
        (*h).push_tag(tag);
    }
}

/// # Safety
/// `h` must be valid; `sizes`/`strides` must have `ndim` elements each (or
/// `strides` may be null when the tensor is not strided).
#[no_mangle]
pub unsafe extern "C" fn eprof_op_inputs_push_tensor(
    h: *mut OpInputs,
    dtype: *const c_char,
    strided: i32,
    ndim: usize,
    sizes: *const i64,
    strides: *const i64,
) {
    if h.is_null() {
        return;
    }
    let empty: [i64; 0] = [];
    let sizes = if sizes.is_null() || ndim == 0 {
        &empty[..]
    } else {
        std::slice::from_raw_parts(sizes, ndim)
    };
    let strides = if strides.is_null() || ndim == 0 || strided == 0 {
        &empty[..]
    } else {
        std::slice::from_raw_parts(strides, ndim)
    };
    (*h).push_tensor(read(dtype), strided != 0, sizes, strides);
}

/// # Safety
/// `h` must be valid; `rendered` NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn eprof_op_inputs_push_scalar(
    h: *mut OpInputs,
    tag: u8,
    rendered: *const c_char,
) {
    if !h.is_null() {
        (*h).push_scalar(tag, read(rendered));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(inputs: &mut OpInputs) -> Vec<(String, String)> {
        inputs.next_metadata()
    }

    fn get<'a>(m: &'a [(String, String)], key: &str) -> &'a str {
        m.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str()).unwrap()
    }

    #[test]
    fn a_tensors_sizes_and_strides_render_separately() {
        let mut o = OpInputs::default();
        o.push_tensor("float".into(), true, &[2, 3], &[3, 1]);
        o.push_tag(TAG_TERMINATOR);
        let m = meta(&mut o);
        assert_eq!(get(&m, "Input Dims"), "[[2, 3]]");
        assert_eq!(get(&m, "Input Strides"), "[[3, 1]]");
        assert_eq!(get(&m, "Input type"), r#"["float"]"#);
    }

    #[test]
    fn a_non_strided_tensor_records_no_strides() {
        let mut o = OpInputs::default();
        o.push_tensor("float".into(), false, &[4], &[]);
        o.push_tag(TAG_TERMINATOR);
        let m = meta(&mut o);
        assert_eq!(get(&m, "Input Dims"), "[[4]]");
        assert_eq!(get(&m, "Input Strides"), "[[]]");
    }

    #[test]
    fn an_input_with_no_shape_still_occupies_a_slot() {
        // The fields are positional -- dropping an entry would shift every
        // later input onto the wrong argument.
        let mut o = OpInputs::default();
        o.push_tag(TAG_OTHER);
        o.push_tensor("long".into(), true, &[5], &[1]);
        o.push_tag(TAG_UNDEFINED_TENSOR);
        o.push_tag(TAG_TERMINATOR);
        let m = meta(&mut o);
        assert_eq!(get(&m, "Input Dims"), "[[], [5], []]");
        assert_eq!(get(&m, "Input type"), r#"["", "long", ""]"#);
    }

    #[test]
    fn a_scalar_carries_its_value_and_no_shape() {
        let mut o = OpInputs::default();
        o.push_scalar(TAG_SCALAR, "1".into());
        o.push_scalar(TAG_SCALAR_LIST, "[2, 3]".into());
        o.push_tag(TAG_TERMINATOR);
        let m = meta(&mut o);
        assert_eq!(get(&m, "Input type"), r#"["Scalar", "ScalarList"]"#);
        assert_eq!(get(&m, "Concrete Inputs"), r#"["1", "[2, 3]"]"#);
        assert_eq!(get(&m, "Input Dims"), "[[], []]");
    }

    #[test]
    fn a_tensor_list_renders_as_a_list_of_shapes() {
        let mut o = OpInputs::default();
        o.push_tag(TAG_TENSOR_LIST_BEGIN);
        o.push_tensor("float".into(), true, &[1], &[1]);
        o.push_tensor("float".into(), true, &[2], &[1]);
        o.push_tag(TAG_TERMINATOR); // ends the list
        o.push_tag(TAG_TERMINATOR); // ends the op
        let m = meta(&mut o);
        assert_eq!(get(&m, "Input Dims"), "[[[1], [2]]]");
        assert_eq!(get(&m, "Input type"), r#"["TensorList"]"#);
    }

    #[test]
    fn one_undefined_member_drops_the_whole_lists_shape() {
        let mut o = OpInputs::default();
        o.push_tag(TAG_TENSOR_LIST_BEGIN);
        o.push_tensor("float".into(), true, &[1], &[1]);
        o.push_tag(TAG_UNDEFINED_TENSOR);
        o.push_tag(TAG_TERMINATOR);
        o.push_tag(TAG_TERMINATOR);
        let m = meta(&mut o);
        assert_eq!(get(&m, "Input Dims"), "[[]]");
        assert_eq!(get(&m, "Input type"), r#"[""]"#);
    }

    #[test]
    fn a_long_tensor_list_is_not_spelled_out() {
        let mut o = OpInputs::default();
        o.push_tag(TAG_TENSOR_LIST_BEGIN);
        for _ in 0..TENSOR_LIST_DISPLAY_LENGTH_LIMIT + 1 {
            o.push_tensor("float".into(), true, &[1], &[1]);
        }
        o.push_tag(TAG_TERMINATOR);
        o.push_tag(TAG_TERMINATOR);
        assert_eq!(get(&meta(&mut o), "Input Dims"), "[[]]");
    }

    #[test]
    fn ops_are_read_back_one_at_a_time_in_order() {
        let mut o = OpInputs::default();
        o.push_tensor("float".into(), true, &[1], &[1]);
        o.push_tag(TAG_TERMINATOR);
        o.push_tensor("long".into(), true, &[2], &[1]);
        o.push_tag(TAG_TERMINATOR);
        assert_eq!(get(&meta(&mut o), "Input Dims"), "[[1]]");
        assert_eq!(get(&meta(&mut o), "Input Dims"), "[[2]]");
        assert!(meta(&mut o).is_empty(), "and then it runs out");
    }

    #[test]
    fn an_op_with_no_inputs_gets_no_metadata_at_all() {
        let mut o = OpInputs::default();
        o.push_tag(TAG_TERMINATOR);
        assert!(meta(&mut o).is_empty());
    }
}
