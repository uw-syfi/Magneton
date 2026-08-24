// See backend.h: one libtorch registration lookup, for the Rust kineto layer.

#include "backend.h"

#include <c10/core/Device.h>

#include <string>

extern "C" {

int32_t eprof_privateuse1_backend_registered(void) {
  return c10::get_privateuse1_backend() != "privateuseone" ? 1 : 0;
}

}  // extern "C"
