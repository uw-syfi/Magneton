#pragma once


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
