#include <string.h>
#include "accel.h"

// Array utilities
TEMPLATE_START(UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, FLOAT16, FLOAT32, FLOAT64)
#undef TEMPLATE_TYPE
#define TEMPLATE_TYPE int32_t
#undef TEMPLATE_NAME
#define TEMPLATE_NAME Int32

void connx_TEMPLATE_NAME_add(int32_t count, TEMPLATE_TYPE* c, TEMPLATE_TYPE* a, TEMPLATE_TYPE* b) {
    for(int32_t i = 0; i < count; i++) {
        c[i] = a[i] + b[i];
    }
}

void connx_TEMPLATE_NAME_sub(int32_t count, TEMPLATE_TYPE* c, TEMPLATE_TYPE* a, TEMPLATE_TYPE* b) {
    for(int32_t i = 0; i < count; i++) {
        c[i] = a[i] - b[i];
    }
}

void connx_TEMPLATE_NAME_mul(int32_t count, TEMPLATE_TYPE* c, TEMPLATE_TYPE* a, TEMPLATE_TYPE* b) {
    for(int32_t i = 0; i < count; i++) {
        c[i] = a[i] * b[i];
    }
}

void connx_TEMPLATE_NAME_broadcast(int32_t y_count, TEMPLATE_TYPE* y, int32_t x_count, TEMPLATE_TYPE* x) {
    for(int32_t i = 0; i < y_count / x_count; i++) {
        memcpy(y + i * sizeof(TEMPLATE_TYPE), x, sizeof(TEMPLATE_TYPE) * x_count);
    }
}

int32_t connx_TEMPLATE_NAME_argmax(int32_t count, TEMPLATE_TYPE* y, TEMPLATE_TYPE* x) {
    int32_t argmax = -1;
    TEMPLATE_TYPE max;

    for(int32_t i = 0; i < count; i++) {
        if(argmax == -1 || x[i] > max) {
            argmax = i;
            max = x[i];
        }
    }

    if(y != NULL) {
        *y = max;
    }

    return argmax;
}

int32_t connx_TEMPLATE_NAME_argmin(int32_t count, TEMPLATE_TYPE* y, TEMPLATE_TYPE* x) {
    int32_t argmin = -1;
    TEMPLATE_TYPE min;

    for(int32_t i = 0; i < count; i++) {
        if(argmin == -1 || x[i] < min) {
            argmin = i;
            min = x[i];
        }
    }

    if(y != NULL) {
        *y = min;
    }

    return argmin;
}

TEMPLATE_TYPE connx_TEMPLATE_NAME_sum(int32_t count, TEMPLATE_TYPE* array) {
    TEMPLATE_TYPE result = 0;

    for(int32_t i = 0; i < count; i++) {
        result += array[i];
    }

    return result;
}

TEMPLATE_TYPE connx_TEMPLATE_NAME_product(int32_t count, TEMPLATE_TYPE* array) {
    TEMPLATE_TYPE result = 1;

    for(int32_t i = 0; i < count; i++) {
        result *= array[i];
    }

    return result;
}
TEMPLATE_END()

// TODO: Implement basic function sfor STRING, BOOL, COMPLEX64, COMPLEX128
