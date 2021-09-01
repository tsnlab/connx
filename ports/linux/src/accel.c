/*
 *  CONNX, C implementation of Open Neural Network Exchange Runtime
 *  Copyright (C) 2019-2021 TSN Lab, Inc.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <float.h>
#include <string.h>

#include <connx/accel.h>

#define CONNX_INT8_MIN INT8_MIN
#define CONNX_INT8_MAX INT8_MAX
#define CONNX_INT16_MIN INT16_MIN
#define CONNX_INT16_MAX INT16_MAX
#define CONNX_INT32_MIN INT32_MIN
#define CONNX_INT32_MAX INT32_MAX
#define CONNX_INT64_MIN INT64_MIN
#define CONNX_INT64_MAX INT64_MAX
#define CONNX_UINT8_MIN 0
#define CONNX_UINT8_MAX UINT8_MAX
#define CONNX_UINT16_MIN 0
#define CONNX_UINT16_MAX UINT16_MAX
#define CONNX_UINT32_MIN 0
#define CONNX_UINT32_MAX UINT32_MAX
#define CONNX_UINT64_MIN 0
#define CONNX_UINT64_MAX UINT64_MAX
#define CONNX_FLOAT16_MIN -65504
#define CONNX_FLOAT16_MAX 65504
#define CONNX_FLOAT32_MIN -FLT_MAX
#define CONNX_FLOAT32_MAX FLT_MAX
#define CONNX_FLOAT64_MIN -DBL_MAX
#define CONNX_FLOAT64_MAX DBL_MAX

// Array utilities
TEMPLATE_START(UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, FLOAT16, FLOAT32, FLOAT64)
#undef TEMPLATE_TYPE
#define TEMPLATE_TYPE int32_t
#undef TEMPLATE_NAME
#define TEMPLATE_NAME Int32
#define TEMPLATE_DTYPE_MAX CONNX_INT32_MAX
#define TEMPLATE_DTYPE_MIN CONNX_INT32_MIN

void connx_TEMPLATE_NAME_add(int32_t count, TEMPLATE_TYPE* c, TEMPLATE_TYPE* a, TEMPLATE_TYPE* b) {
    for (int32_t i = 0; i < count; i++) {
        c[i] = a[i] + b[i];
    }
}

void connx_TEMPLATE_NAME_sub(int32_t count, TEMPLATE_TYPE* c, TEMPLATE_TYPE* a, TEMPLATE_TYPE* b) {
    for (int32_t i = 0; i < count; i++) {
        c[i] = a[i] - b[i];
    }
}

void connx_TEMPLATE_NAME_mul(int32_t count, TEMPLATE_TYPE* c, TEMPLATE_TYPE* a, TEMPLATE_TYPE* b) {
    for (int32_t i = 0; i < count; i++) {
        c[i] = a[i] * b[i];
    }
}

TEMPLATE_TYPE connx_TEMPLATE_NAME_mul_and_sum(int32_t count, TEMPLATE_TYPE* a, TEMPLATE_TYPE* b) {
    TEMPLATE_TYPE sum = 0;

    for (int32_t i = 0; i < count; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

void connx_TEMPLATE_NAME_broadcast(int32_t y_count, TEMPLATE_TYPE* y, int32_t x_count, TEMPLATE_TYPE* x) {
    for (int32_t i = 0; i < y_count / x_count; i++) {
        memcpy(y + i, x, sizeof(TEMPLATE_TYPE) * x_count);
    }
}

int32_t connx_TEMPLATE_NAME_argmax(int32_t count, TEMPLATE_TYPE* y, TEMPLATE_TYPE* x) {
    int32_t argmax = -1;
    TEMPLATE_TYPE max = TEMPLATE_DTYPE_MIN;

    for (int32_t i = 0; i < count; i++) {
        if (argmax == -1 || x[i] > max) {
            argmax = i;
            max = x[i];
        }
    }

    if (y != NULL) {
        *y = max;
    }

    return argmax;
}

int32_t connx_TEMPLATE_NAME_argmin(int32_t count, TEMPLATE_TYPE* y, TEMPLATE_TYPE* x) {
    int32_t argmin = -1;
    TEMPLATE_TYPE min = TEMPLATE_DTYPE_MAX;

    for (int32_t i = 0; i < count; i++) {
        if (argmin == -1 || x[i] < min) {
            argmin = i;
            min = x[i];
        }
    }

    if (y != NULL) {
        *y = min;
    }

    return argmin;
}

TEMPLATE_TYPE connx_TEMPLATE_NAME_sum(int32_t count, TEMPLATE_TYPE* array) {
    TEMPLATE_TYPE result = 0;

    for (int32_t i = 0; i < count; i++) {
        result += array[i];
    }

    return result;
}

TEMPLATE_TYPE connx_TEMPLATE_NAME_product(int32_t count, TEMPLATE_TYPE* array) {
    TEMPLATE_TYPE result = 1;

    for (int32_t i = 0; i < count; i++) {
        result *= array[i];
    }

    return result;
}
TEMPLATE_END()

// TODO: Implement basic function sfor STRING, BOOL, COMPLEX64, COMPLEX128
