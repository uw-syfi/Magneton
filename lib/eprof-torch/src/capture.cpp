// Reading an at::RecordFunction on the way in.
//
// This is the read half of the callback: pulling an op's name, scope and inputs
// off the at::RecordFunction, and printing the c10::IValues it carries. It
// decides nothing and stores nothing -- where the results go is
// eprof-torch/src/{run,inputs}.rs.
//
// It runs inside dispatch, once per op, so each entry point answers as much as
// it can per call: EprofOpFields is filled in one crossing rather than by field.

#include "capture.h"

#include "rust_queue.h"

#include <sstream>
#include <string>

#include <c10/core/ScalarType.h>
#include <c10/util/Logging.h>

#include "rust_op_inputs.h"
#include "metadata.h"

namespace {

// A scalar list longer than this is not recorded: the values are kept verbatim,
// and a long one would dwarf the event it describes.
constexpr size_t SCALAR_LIST_LENGTH_LIMIT = 30;

// --------------------------------------------------------------------------
// The queue, its per-thread subqueues and every store under them live in Rust
// (eprof-torch/src/queue.rs). What is left here is the reading: an
// at::RecordFunction on the way in, and the c10::IValues it carries.

// How a scalar reads in the trace: an absent value is empty rather than the
// word "None", matching how a list of concrete inputs is rendered.
std::string PrintScalar(const c10::IValue &value) {
  if (value.isNone()) {
    return {};
  }
  std::stringstream ss;
  ss << value;
  return ss.str();
}

void PushTensor(EprofOpInputs *h, const at::Tensor &t) {
  // TODO fix nested and symbolic sizes
  if (!t.defined() || t.is_nested() ||
      t.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
    eprof_op_inputs_push_tag(h, EPROF_INPUT_UNDEFINED_TENSOR);
    return;
  }
  // Only a strided tensor has strides to record.
  const bool strided = t.layout() == at::kStrided;
  eprof_op_inputs_push_tensor(
      h, std::string(scalarTypeToTypeMeta(t.scalar_type()).name()).c_str(),
      strided, t.sizes().size(), t.sizes().data(),
      strided ? t.strides().data() : nullptr);
}

static bool IsSupportedScalarList(const c10::IValue &list_candidate) {
  if (!list_candidate.isList()) {
    return false;
  }
  auto list_ref = list_candidate.toListRef();
  if (C10_UNLIKELY(list_ref.empty())) {
    return true;
  }
  if (C10_UNLIKELY(!list_ref[0].isScalar())) {
    return false;
  }
  return list_ref.size() <= SCALAR_LIST_LENGTH_LIMIT;
}

static void PushOpInputs(void *inputs, c10::ArrayRef<const c10::IValue> values) {
  auto *h = static_cast<EprofOpInputs *>(inputs);
  for (const auto &value : values) {
    if (value.isTensor()) {
      PushTensor(h, value.toTensor());
    } else if (value.isScalar()) {
      eprof_op_inputs_push_scalar(h, EPROF_INPUT_SCALAR,
                                  PrintScalar(value).c_str());
    } else if (value.isTensorList()) {
      eprof_op_inputs_push_tag(h, EPROF_INPUT_TENSOR_LIST_BEGIN);
      for (const auto &t : value.toTensorList()) {
        PushTensor(h, t);
      }
      eprof_op_inputs_push_tag(h, EPROF_INPUT_TERMINATOR);
    } else if (IsSupportedScalarList(value)) {
      eprof_op_inputs_push_scalar(h, EPROF_INPUT_SCALAR_LIST,
                                  PrintScalar(value).c_str());
    } else {
      eprof_op_inputs_push_tag(h, EPROF_INPUT_OTHER);
    }
  }
  eprof_op_inputs_push_tag(h, EPROF_INPUT_TERMINATOR);
}

// An op's keyword arguments, rendered on the way in. Only scalars are worth
// carrying; anything else would need the IValue to survive until the trace is
// written, which is exactly what recording strings avoids.
void PushKwinputs(EprofSubqueue *sq, const at::RecordFunction &fn) {
  for (const auto &[key, val] : fn.kwinputs()) {
    if (key == "stream" && val.isInt()) {
      eprof_subqueue_add_op_metadata(
          sq, key.c_str(), eprof::ivalueToStr(val, false).c_str(), 1);
      continue;
    }
    if (!val.isString() && !val.isDouble() && !val.isInt() && !val.isBool()) {
      LOG(WARNING) << "Kwinputs' value must be a scalar, but " << key
                   << " is not an int, double, string, or bool for op: "
                   << fn.name() << " skipping";
      continue;
    }
    eprof_subqueue_add_op_metadata(
        sq, key.c_str(), eprof::ivalueToStr(val, val.isString()).c_str(), 1);
  }
}

}  // namespace

extern "C" {

void eprof_rf_read(const void *fn, EprofOpFields *out) {
  const auto &f = *static_cast<const at::RecordFunction *>(fn);
  out->name = f.name();
  out->sequence_number = f.seqNr();
  out->forward_tid = f.forwardThreadId();
  out->scope = static_cast<uint8_t>(f.scope());
  out->record_function_id = f.handle();
  out->is_nccl_meta = f.isNcclMeta() ? 1 : 0;
  out->is_user_scope = f.scope() == at::RecordScope::USER_SCOPE ? 1 : 0;
}

void eprof_rf_push_inputs(const void *fn, void *inputs) {
  PushOpInputs(
      inputs, static_cast<const at::RecordFunction *>(fn)->inputs());
}

void eprof_rf_push_kwinputs(const void *fn, void *subqueue) {
  PushKwinputs(static_cast<EprofSubqueue *>(subqueue),
                               *static_cast<const at::RecordFunction *>(fn));
}

void eprof_rf_push_nccl_meta(const void *fn, void *subqueue) {
  auto *sq = static_cast<EprofSubqueue *>(subqueue);
  for (const auto &[key, value] :
       eprof::saveNcclMeta(*static_cast<const at::RecordFunction *>(fn))) {
    eprof_subqueue_add_op_metadata(sq, key.c_str(), value.c_str(), 0);
  }
}

}  // extern "C"
