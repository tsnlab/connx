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

#define GEMM_DEBUG 0

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
int Gemm_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
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
    bool is_biased = false;

    if (input_count == 3) {
        C = connx_Graph_get(graph, inputs[2]);
        assert(C != NULL);
        is_biased = true;
    }

    // A, B rows/cols alias for transposed flags
    int32_t A_rows = A->shape[0];
    int32_t A_cols = A->shape[1];
    int32_t B_rows = B->shape[0];
    int32_t B_cols = B->shape[1];
    if (transA && transB) {
        A_rows = A->shape[1];
        A_cols = A->shape[0];
        B_rows = B->shape[1];
        B_cols = B->shape[0];
    } else if (transA && !transB) {
        A_rows = A->shape[1];
        A_cols = A->shape[0];
    } else if (!transA && transB) {
        B_rows = B->shape[1];
        B_cols = B->shape[0];
    }

    // A's # of cols and B's # of row should be same
    if (A_cols != B_rows) {
        connx_error("Gemm: A->shape[1] != B->shape[0]");
        return CONNX_TENSOR_SHAPE_NOT_MATCHING;
    }

    // initialize output tensor
    int ndim = A->ndim; // Always 2D
    int32_t shape[] = {A_rows, B_cols};
    connx_Tensor* Y = connx_Tensor_alloc(A->dtype, ndim, shape);
    if (Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

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

    switch (A->dtype) {
        /*{% set type_constraints = [
            FLOAT16, FLOAT32, FLOAT64,
            UINT32, UINT64,
            INT32, INT64,
        ]
        %}*/
        /*{% for DTYPE, TYPE in loop_types(*type_constraints) %}*/
    case {{ DTYPE }}: {
        // clang-format off
        {{TYPE}} bias = 0;
        // clang-format on
        if (is_biased) {
            // for single value for bias
            if (C->ndim == 0 || (C->ndim == 1 && C->shape[0] == 1)) {
                bias = *({{TYPE}}*)C->buffer;
            }
        }
        float32_t(*y)[Y->shape[1]] = Y->buffer;
        // multiplication w/ transposed flag
        for (int32_t row = 0; row < A_rows; ++row) {
            for (int32_t col = 0; col < B_cols; ++col) {
                float32_t sum = 0.0;
                for (int32_t k = 0; k < A_cols; ++k) {
                    // clang-format off
                    {{TYPE}} a_val = transA ? a[k][row] : a[row][k];
                    {{TYPE}} b_val = transB ? b[col][k] : b[k][col];
                    // clang-format on
                    sum += alpha * a_val * b_val;
                    // fprintf(stderr, "a[%d][%d] b[%d][%d]\n", r, k, k, c);
                }
                // fprintf(stderr, "%f\n", sum);
                if (is_biased) {
                    {{TYPE}}(*c)[C->shape[1]] = C->buffer;
                    if (C->ndim == 0 || (C->ndim == 1 && C->shape[0] == 1)) {
                        // already assigned once above, so do nothing
                    } else if (C->ndim == 2 && C->shape[0] == 1) {
                        bias = (({{TYPE}}*)C->buffer)[col];
                    } else {
                        bias = c[row][col];
                    }
                    y[row][col] += sum + beta * bias;
                } else {
                    y[row][col] = sum;
                }
            }
        }
        break;
    }

    /*{% endfor %}*/
    default:
        connx_error("Gemm: Datatype %d is not supported yet.\n", A->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }
#if GEMM_DEBUG
    connx_debug("Y ndim %d ", Y->ndim);
    connx_Tensor_dump(Y);
#endif

    connx_Graph_set(graph, outputs[0], Y);
    return CONNX_OK;
}
