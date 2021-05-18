#ifndef __CONNX_ACCEL_H__
#define __CONNX_ACCEL_H__

#include "types.h"

#define DEFINE_BASIC(NAME, TYPE)                                       \
    void connx_##NAME##_add(int32_t count, TYPE* c, TYPE* a, TYPE* b); \
    void connx_##NAME##_sub(int32_t count, TYPE* c, TYPE* a, TYPE* b); \
    void connx_##NAME##_mul(int32_t count, TYPE* c, TYPE* a, TYPE* b); \
    int32_t connx_##NAME##_argmax(int32_t count, TYPE* y, TYPE* x);    \
    int32_t connx_##NAME##_argmin(int32_t count, TYPE* y, TYPE* x);    \
    TYPE connx_##NAME##_sum(int32_t count, TYPE* array);               \
    TYPE connx_##NAME##_product(int32_t count, TYPE* array);

DEFINE_BASIC(Uint8, uint8_t)
DEFINE_BASIC(Int8, int8_t)
DEFINE_BASIC(Uint16, uint16_t)
DEFINE_BASIC(Int16, int16_t)
DEFINE_BASIC(Uint32, uint32_t)
DEFINE_BASIC(Int32, int32_t)
DEFINE_BASIC(Uint64, uint64_t)
DEFINE_BASIC(Int64, int64_t)
DEFINE_BASIC(Float16, float16_t)
DEFINE_BASIC(Float32, float32_t)
DEFINE_BASIC(Float64, float64_t)
DEFINE_BASIC(String, char*)
DEFINE_BASIC(Bool, bool)
DEFINE_BASIC(Complex64, void*)
DEFINE_BASIC(Complex128, void*)

#endif /* __CONNX_ACCEL_H__ */
