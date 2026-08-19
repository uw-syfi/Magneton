#pragma once

// Rendering an op's metadata out of libtorch types.
//
// The values themselves live on the Rust event; what is here is the reading:
// printing a c10::IValue, and pulling a collective's description out of the
// thread-local ParamCommsDebugInfo that c10d leaves behind.

#include <ATen/record_function.h>
#include <ATen/core/ivalue.h>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace eprof {

std::string ivalueToStr(const c10::IValue &val, bool isString);

// The collective-communication metadata of an nccl op. Long rank and split
// lists are shortened; the full ones would dwarf the event.
std::unordered_map<std::string, std::string> saveNcclMeta(
    const at::RecordFunction &fn);

}  // namespace eprof
