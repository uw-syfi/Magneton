// C ABI for the python tracer's caches (eprof-python/src/cache.rs).
//
// The tracer keys a callsite on cheap identities -- interned string pointers
// and PyObject addresses -- and only reads the descriptive strings out of
// CPython the first time it sees one. This header is the boundary: C++ does
// the CPython reads, Rust owns every map.

#ifndef EPROF_RUST_PY_CACHE_H
#define EPROF_RUST_PY_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EprofPyCache EprofPyCache;

// Call kinds. A module call is still a python call in the event stream.
#define EPROF_PY_CALL 0
#define EPROF_PY_MODULE_CALL 1
#define EPROF_PY_C_CALL 2

// A python code location, identified by the identity of its interned filename
// and name strings plus the line -- comparing pointers is what keeps this
// usable on the tracer's hot path.
typedef struct {
  uint64_t filename;
  uint64_t name;
  int32_t line;
} EprofCodeLoc;

// What gets interned. `value_loc` identifies a plain python call; `value_ptr`
// the kinds keyed on a PyObject instead (module instance, bound C function).
// `python_tid` is part of the key so two threads at one callsite get two keys.
typedef struct {
  uint8_t call_type;
  uint64_t python_tid;
  EprofCodeLoc value_loc;
  uint64_t value_ptr;
  EprofCodeLoc caller;
} EprofSiteKey;

// A resolved callsite. The strings point into the cache and stay valid until
// it is destroyed; they are never null, only empty.
typedef struct {
  uint64_t key;
  int32_t event_type;
  uint64_t python_tid;
  int32_t caller_line;
  const char *caller_filename;
  const char *caller_name;
  int32_t callsite_line;
  const char *callsite_filename;
  const char *callsite_name;
  int32_t has_module;
  const char *module_cls_name;
  uint64_t module_id;
  const char *function_name;
} EprofPySite;

EprofPyCache *eprof_pycache_create(void);
void eprof_pycache_destroy(EprofPyCache *cache);

// Writes `nn.Module.__call__`'s location; 0 if it has not been seen yet.

// 0 when the site has not been interned, which is the signal to go do the
// CPython reads and then call intern.
uint64_t eprof_pycache_get(const EprofPyCache *cache, const EprofSiteKey *key);
/* Everything the cache might need to describe a callsite it has not seen. The
 * tracer fills this once, on a miss; the cache uses only the parts it lacks.
 * Cheaper than asking first -- a miss is once per callsite for the whole run,
 * so one crossing answers all of them. */
typedef struct {
  EprofSiteKey key;
  const char *caller_filename;
  const char *caller_funcname;
  /* A plain call's own location, or nn.Module.__call__'s for a module call. */
  const char *value_filename;
  const char *value_funcname;
  /* Module calls only. */
  EprofCodeLoc module_loc;
  uint64_t module_cls;
  const char *module_cls_name;
  /* C calls only. */
  const char *c_function_name;
} EprofSiteRecord;

uint64_t eprof_pycache_intern_site(EprofPyCache *cache,
                                   const EprofSiteRecord *rec);

void eprof_pycache_trim_prefixes(EprofPyCache *cache, const char *const *prefixes,
                                 size_t n);

// One recorded entry into a frame. `system_tid` is EPROF_NO_TID for a frame
// that was already on the stack when profiling started.
#define EPROF_NO_TID UINT64_MAX
typedef struct {
  uint64_t key;
  uint64_t system_tid;
  int32_t device;
  int32_t resource;
  int64_t start_ns;
} EprofEnterRecord;

// One recorded exit. Which frame it closes is worked out by the replay.
typedef struct {
  int64_t t;
  uint64_t python_tid;
  int32_t is_c_call;
} EprofExitRecord;

// Pairs enters with exits and pushes the resulting events onto the array.
// Sorts `enters` in place.
void eprof_pycache_replay(EprofPyCache *cache, void *events,
                          EprofEnterRecord *enters, size_t n_enters,
                          const EprofExitRecord *exits, size_t n_exits,
                          int64_t end_time_ns);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EPROF_RUST_PY_CACHE_H
