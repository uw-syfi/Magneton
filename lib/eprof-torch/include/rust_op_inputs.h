/* An op's recorded inputs (eprof-torch/src/inputs.rs).
 *
 * Capture pushes one group per op, terminated by EPROF_INPUT_TERMINATOR, and
 * materialization reads the groups back in the same order to render the
 * "Input Dims" / "Input Strides" / "Input type" / "Concrete Inputs" fields.
 *
 * C++ keeps only what needs an at::Tensor or a c10::IValue: choosing an input's
 * tag, reading a tensor's dtype and sizes, and printing a scalar.
 */

#ifndef EPROF_RUST_OP_INPUTS_H
#define EPROF_RUST_OP_INPUTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EprofOpInputs EprofOpInputs;

/* Input kinds. A tensor list is EPROF_INPUT_TENSOR_LIST_BEGIN, its members,
 * then a terminator of its own. */
#define EPROF_INPUT_TENSOR 0
#define EPROF_INPUT_UNDEFINED_TENSOR 1
#define EPROF_INPUT_TENSOR_LIST_BEGIN 2
#define EPROF_INPUT_SCALAR_LIST 3
#define EPROF_INPUT_SCALAR 4
#define EPROF_INPUT_OTHER 5
#define EPROF_INPUT_TERMINATOR 6

/* For the kinds that carry no payload. */
void eprof_op_inputs_push_tag(EprofOpInputs *h, uint8_t tag);

/* Pushes a tensor input. `strides` is read only when `strided` is set -- a
 * tensor with another layout has none. */
void eprof_op_inputs_push_tensor(EprofOpInputs *h, const char *dtype,
                                 int32_t strided, size_t ndim,
                                 const int64_t *sizes, const int64_t *strides);

/* Pushes a scalar or scalar list, already printed. */
void eprof_op_inputs_push_scalar(EprofOpInputs *h, uint8_t tag,
                                 const char *rendered);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EPROF_RUST_OP_INPUTS_H */
