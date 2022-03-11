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
#ifdef __SSE__
#include <immintrin.h>
#endif /* __SSE__ */

#include <string.h>

#include <connx/accel.h>

// Ref: https://newbedev.com/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
#ifdef __SSE3__
float hsum_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif /* __SSE3__ */

#ifdef __AVX__
float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow = _mm_add_ps(vlow, vhigh);             // add the low 128
    return hsum_ps(vlow);                       // and inline the sse3 version, which is optimal for AVX
    // (no wasted instructions, and all of them are the 4B minimum)
}
#endif /* __AVX__ */

// Array utilities
/*{% for DTYPE, TYPE in loop_types(
UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, FLOAT16, FLOAT32, FLOAT64) %}*/

// clang-format off
void connx_{{ DTYPE | to_name }}_add(int32_t count, {{TYPE}}* c, {{TYPE}}* a, {{TYPE}}* b) {
    // clang-format on
    for (int32_t i = 0; i < count; i++) {
        c[i] = a[i] + b[i];
    }
}

// clang-format off
void connx_{{ DTYPE | to_name }}_sub(int32_t count, {{TYPE}}* c, {{TYPE}}* a, {{TYPE}}* b) {
    // clang-format on
    for (int32_t i = 0; i < count; i++) {
        c[i] = a[i] - b[i];
    }
}

// clang-format off
void connx_{{ DTYPE | to_name }}_mul(int32_t count, {{TYPE}}* c, {{TYPE}}* a, {{TYPE}}* b) {
    // clang-format on
    for (int32_t i = 0; i < count; i++) {
        c[i] = a[i] * b[i];
    }
}

// clang-format off
{{TYPE}} connx_{{ DTYPE | to_name }}_mul_and_sum(int32_t count, {{TYPE}}* a, {{TYPE}}* b) {
    {{TYPE}} sum = 0;
    // clang-format on

    for (int32_t i = 0; i < count; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

// clang-format off
void connx_{{ DTYPE | to_name }}_broadcast(int32_t y_count, {{TYPE}}* y, int32_t x_count, {{TYPE}}* x) {
    // clang-format on
    for (int32_t i = 0; i < y_count / x_count; i++) {
        memcpy(y + i, x, sizeof({{TYPE}}) * x_count);
    }
}

// clang-format off
int32_t connx_{{ DTYPE | to_name }}_argmax(int32_t count, {{TYPE}}* y, {{TYPE}}* x) {
    int32_t argmax = -1;
    {{TYPE}} max = CONNX_{{ DTYPE }}_MIN;
    // clang-format on

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

// clang-format off
int32_t connx_{{ DTYPE | to_name }}_argmin(int32_t count, {{TYPE}}* y, {{TYPE}}* x) {
    int32_t argmin = -1;
    {{TYPE}} min = CONNX_{{ DTYPE }}_MAX;
    // clang-format on

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

// clang-format off
{{TYPE}} connx_{{ DTYPE | to_name }}_sum(int32_t count, {{TYPE}}* array) {
    {{TYPE}} result = 0;
    // clang-format on

    for (int32_t i = 0; i < count; i++) {
        result += array[i];
    }

    return result;
}

// clang-format off
{{TYPE}} connx_{{ DTYPE | to_name }}_product(int32_t count, {{TYPE}}* array) {
    {{TYPE}} result = 1;
    // clang-format on

    for (int32_t i = 0; i < count; i++) {
        result *= array[i];
    }

    return result;
}
/*{% endfor %}*/

// TODO: Implement basic function sfor STRING, BOOL, COMPLEX64, COMPLEX128
