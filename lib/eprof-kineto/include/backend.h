/* The one libtorch question the kineto layer has to ask.
 *
 * Kineto reports an activity's device by activity type, and a custom backend
 * registered on the PrivateUse1 slot reuses the CUDA and MTIA types -- so the
 * type alone does not say which device ran it. Whether such a backend exists
 * is a libtorch registration, visible only from C++.
 */
#ifndef EPROF_KINETO_BACKEND_H
#define EPROF_KINETO_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Nonzero if something has claimed the PrivateUse1 device slot. */
int32_t eprof_privateuse1_backend_registered(void);

#ifdef __cplusplus
}
#endif

#endif /* EPROF_KINETO_BACKEND_H */
