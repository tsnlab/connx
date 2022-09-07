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

    // alias for A, B rows/cols whether transposed flags enabled
    int32_t A_rows = A->shape[0];
    int32_t A_cols = A->shape[1];
    int32_t B_rows = B->shape[0];
    int32_t B_cols = B->shape[1];

    if (transA) {
        A_rows = A->shape[1];
        A_cols = A->shape[0];
    }

    if (transB) {
        B_rows = B->shape[1];
        B_cols = B->shape[0];
    }

    // A's # of cols and B's # of row should be same
    if (A_cols != B_rows) {
        connx_error("Gemm: A_cols != B_rows");
        return CONNX_TENSOR_SHAPE_NOT_MATCHING;
    }

    int32_t shape[] = {A_rows, B_cols};
    connx_Tensor* C = NULL;
    bool is_biased = false;
    if (input_count == 3) {
        C = connx_Graph_get(graph, inputs[2]);
        assert(C != NULL);
        // Check if Unidirectional Broadcasting is possible,
        if (C->ndim == 0) { // scalar type
            is_biased = true;
        } else if (C->ndim == 1) {
            if (C->shape[0] == 1 || C->shape[0] == shape[1]) {
                is_biased = true;
            }
        } else if (C->ndim == 2) {
            if ((C->shape[0] == 1 || C->shape[0] == shape[0]) && (C->shape[1] == 1 || C->shape[1] == shape[1])) {
                is_biased = true;
            }
        }

        if (!is_biased) {
            connx_error("Gemm: Unidirectional Broadcasting is not possible");
            return CONNX_TENSOR_SHAPE_NOT_MATCHING;
        }
    }

    // initialize output tensor
    int ndim = 2; // Always 2D
    connx_Tensor* Y = connx_Tensor_alloc(A->dtype, ndim, shape);
    if (Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

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
        {{TYPE}}(*y)[Y->shape[1]] = Y->buffer;
        // multiplication w/ transposed flag
        for (int32_t row = 0; row < A_rows; ++row) {
            for (int32_t col = 0; col < B_cols; ++col) {
                // clang-format off
                {{TYPE}} sum = 0.0;
                // clang-format on
                for (int32_t k = 0; k < A_cols; ++k) {
                    // clang-format off
                    {{TYPE}} a_val = transA ? a[k][row] : a[row][k];
                    {{TYPE}} b_val = transB ? b[col][k] : b[k][col];
                    // clang-format on
                    sum += a_val * b_val;
                }

                if (is_biased) {
                    {{TYPE}}(*c)[C->shape[1]] = C->buffer;
                    if (C->ndim == 0 || (C->ndim == 1 && C->shape[0] == 1) ||
                        (C->ndim == 2 && C->shape[0] == 1 && C->shape[1] == 1)) {
                        // already assigned for single value once above, so do nothing
                    } else if (C->ndim == 1 && C->shape[0] > 1) {
                        bias = (({{TYPE}}*)C->buffer)[col];
                    } else if (C->ndim == 2 && C->shape[0] == 1) {
                        bias = (({{TYPE}}*)C->buffer)[col];
                    } else if (C->ndim == 2 && C->shape[1] == 1) {
                        bias = (({{TYPE}}*)C->buffer)[row];
                    } else {
                        bias = c[row][col];
                    }
                    y[row][col] += alpha * sum + beta * bias;
                } else {
                    y[row][col] = alpha * sum;
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

    connx_Graph_set(graph, outputs[0], Y);
    return CONNX_OK;
}
