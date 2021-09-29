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

#if TEMPLATE_DTYPE == 1 // if FLOAT32 CONNX(alive)
#ifdef __AVX2__         // CONNX(alive)
    while (count >= 16) {
        __m512 va = _mm512_loadu_ps(a);
        __m512 vb = _mm512_loadu_ps(b);

        __m512 vc = _mm512_mul_ps(va, vb);

        // sum += hsum512_ps(vc); TODO implement it
        float c[16];
        _mm512_storeu_ps(c, vc);

        __m256 vc1 = _mm256_loadu_ps(c);
        __m256 vc2 = _mm256_loadu_ps(c + 8);

        sum += hsum256_ps(vc1);
        sum += hsum256_ps(vc2);

        count -= 16;
        a += 16;
        b += 16;
    }
#endif /* __AVX2__ */ // CONNX(alive)

#ifdef __AVX__ // CONNX(alive)
    while (count >= 8) {
        __m256 va = _mm256_loadu_ps(a);
        __m256 vb = _mm256_loadu_ps(b);

        __m256 vc = _mm256_mul_ps(va, vb);

        sum += hsum256_ps(vc);

        count -= 8;
        a += 8;
        b += 8;
    }
#endif /* __AVX__ */ // CONNX(alive)

#ifdef __SSE3__ // CONNX(alive)
    while (count >= 4) {
        __m128 va = _mm_loadu_ps(a);
        __m128 vb = _mm_loadu_ps(b);

        __m128 vc = _mm_mul_ps(va, vb);

        sum += hsum_ps(vc);

        count -= 4;
        a += 4;
        b += 4;
    }
#endif /* __SSE3__ */            // CONNX(alive)
#endif /* TEMPLATE_DTYPE == 1 */ // CONNX(alive)

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
