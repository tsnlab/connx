/*
 *  CONNX, C implementation of Open Neural Network Exchange Runtime
 *  Copyright (C) 2019-2022 TSN Lab, Inc.
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

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <connx/accel.h>
#include <connx/connx.h>

#include "connx/types.h"

#define GEMM_DEBUG 1

__attribute__((unused)) static void show_elements_tensor(char* name, int32_t rows, int32_t cols, float32_t* d,
                                                         bool transposed) {
    fprintf(stderr, "%s %s Transposed ( %d, %d)\n", name, transposed ? "" : "Not", rows, cols);
    int32_t num = transposed ? rows : cols;
    float32_t(*m)[num] = (float32_t(*)[num])d;
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t c = 0; c < cols; c++) {
            float32_t val = transposed ? m[c][r] : m[r][c];
            fprintf(stderr, "%f ", val);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

// clang-format off
// int Gemm_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
//                          // clang-format on
//                          __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
//                          __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    int32_t Gemm_1(connx_Graph * graph, __attribute__((unused)) uint32_t output_count, uint32_t * outputs,
               // clang-format on
               __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
               __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {

    connx_Tensor* A = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* B = connx_Graph_get(graph, inputs[1]);

    assert(A != NULL && B != NULL);

    float32_t alpha = *(float32_t*)attributes[0];
    float32_t beta = *(float32_t*)attributes[1];
    bool transA = *(int32_t*)attributes[2] ? true : false;
    bool transB = *(int32_t*)attributes[3] ? true : false;

    float32_t(*a)[A->shape[1]] = A->buffer;
    float32_t(*b)[B->shape[1]] = B->buffer;

    connx_Tensor* C = NULL;
    if (input_count == 3) {
        C = connx_Graph_get(graph, inputs[2]);
        assert(C != NULL);
    }

    // Transpose A, b checks
    int A_rows;
    int A_cols;
    int B_rows;
    int B_cols;
    if (transA && transB) {
        A_rows = A->shape[1];
        A_cols = A->shape[0];
        B_rows = B->shape[1];
        B_cols = B->shape[0];
    } else if (transA && !transB) {
        A_rows = A->shape[1];
        A_cols = A->shape[0];
        B_rows = B->shape[0];
        B_cols = B->shape[1];
    } else if (!transA && transB) {
        A_rows = A->shape[0];
        A_cols = A->shape[1];
        B_rows = B->shape[1];
        B_cols = B->shape[0];
    } else {
        A_rows = A->shape[0];
        A_cols = A->shape[1];
        B_rows = B->shape[0];
        B_cols = B->shape[1];
    }

    // A's # of cols and B's # of row should be same
    if (A_cols != B_rows) {
        connx_error("Gemm: A->shape[1] != B->shape[0]");
        return CONNX_TENSOR_SHAPE_NOT_MATCHING;
    }

    // initialize output tensor
    int ndim = A->ndim; // Always 2D
    int32_t shape[ndim];
    shape[0] = A_rows;
    shape[1] = B_cols;
    connx_Tensor* Y = connx_Tensor_alloc(A->dtype, ndim, shape);
    if (Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }
    __attribute__((unused)) float32_t(*y)[Y->shape[1]] = Y->buffer;

#if GEMM_DEBUG // TODO: erase this code
    fprintf(stderr, "\n");
    connx_debug("Gemm: Y <- 𝛼AB + βC\n");
    connx_debug("Gemm input_count: %d, attribute_count %d\n", input_count, attribute_count);
    connx_debug("Gemm: alpha %f, beta %f, transA %d, transB %d\n\n", alpha, beta, transA, transB);
    connx_debug("A ndim %d ", A->ndim);
    connx_Tensor_dump(A);
    connx_debug("B ndim %d ", B->ndim);
    connx_Tensor_dump(B);
    if (C != NULL) {
        connx_debug("C ndim %d ", C->ndim);
        connx_Tensor_dump(C);
    }
    connx_debug("Gemm: A_rows %d, A_cols %d, B_rows %d, B_cols %d\n", A_rows, A_cols, B_rows, B_cols);

    if (transA) {
        show_elements_tensor("Tensor A", A_rows, A_cols, A->buffer, transA);
    }
    if (transB) {
        show_elements_tensor("Tensor B", B_rows, B_cols, B->buffer, transB);
    }
#endif

    // multiplication w/ transposed flag
    for (int32_t r = 0; r < A_rows; ++r) {     // from A row 0 to row A->shape[0]
        for (int32_t c = 0; c < B_cols; ++c) { // from B col 0 to col B->shape[1]
            float32_t sum = 0.0;
            for (int32_t k = 0; k < A_cols; ++k) { // Repeat * & sum as A # of cols == B # of rows
                float32_t a_val = transA ? a[k][r] : a[r][k];
                float32_t b_val = transB ? b[c][k] : b[k][c];
                //  float32_t a_val = a[i][k];
                //  float32_t b_val = b[k][j];
                sum += alpha * a_val * b_val; // TODO: + bC
                fprintf(stderr, "a[%d][%d] b[%d][%d]\n", r, k, k, c);
            }
            fprintf(stderr, "%f\n", sum);
            y[r][c] = sum;
        }
    }

    // bias
    if (C != NULL) {
        float32_t bias = 0.0;
        bool single_value_bias = false;
        switch (C->ndim) {
        case 0:
            single_value_bias = true;
            break;
        case 1:
            if (C->shape[0] == 1) {
                single_value_bias = true;
            }
            break;
        default:
            bias = 0.0;
        }
        if (single_value_bias)
            bias = *(float32_t*)C->buffer;

        float32_t(*c)[C->shape[1]] = C->buffer;
        for (int32_t i = 0; i < Y->shape[0]; ++i) {
            for (int32_t j = 0; j < Y->shape[1]; ++j) {
                if (!single_value_bias) {
                    if (C->ndim == 2 && C->shape[0] == 1) {
                        bias = ((float32_t*)C->buffer)[j];
                    } else {
                        bias = c[i][j];
                    }
                }
                y[i][j] += beta * bias;
                fprintf(stderr, "beta %f bias %f orig value %f single? %d\n", beta, bias, y[i][j], single_value_bias);
            }
        }
    }

#if GEMM_DEBUG
    connx_debug("Y ndim %d ", Y->ndim);
    connx_Tensor_dump(Y);
#endif

    connx_Graph_set(graph, outputs[0], Y);
    return CONNX_OK;
}
