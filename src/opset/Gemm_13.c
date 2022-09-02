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

static void show_elements_normal(int32_t rows, int32_t cols, float32_t (*m)[cols]) {

    fprintf(stderr, "Not Transposed ( %d, %d)\n", rows, cols);

    for (int32_t r = 0; r < rows; r++) {
        for (int32_t c = 0; c < cols; c++) {
            float32_t val = m[r][c];
            fprintf(stderr, "%f ", val);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

static void show_elements_transposed(int32_t rows, int32_t cols, float32_t (*m)[rows]) {

    fprintf(stderr, "Transposed ( %d, %d)\n", rows, cols);

    for (int32_t r = 0; r < rows; r++) {
        for (int32_t c = 0; c < cols; c++) {
            float32_t val = m[c][r];
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

    connx_debug("Gemm: Y <- 𝛼AB + βC\n");
    connx_debug("Gemm input_count: %d, attribute_count %d\n", input_count, attribute_count);

    connx_Tensor* A = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* B = connx_Graph_get(graph, inputs[1]);

    assert(A != NULL && B != NULL);

    float32_t alpha = *(float32_t*)attributes[0];
    float32_t beta = *(float32_t*)attributes[1];
    bool transA = *(int32_t*)attributes[2] ? true : false;
    bool transB = *(int32_t*)attributes[3] ? true : false;

#if 1 // TODO: erase this code
    connx_debug("Gemm: alpha %f, beta %f, transA %d, transB %d\n\n", alpha, beta, transA, transB);
    connx_debug("A ndim %d ", A->ndim);
    connx_Tensor_dump(A);
    connx_debug("B ndim %d ", B->ndim);
    connx_Tensor_dump(B);
#endif

    float32_t(*a)[A->shape[1]] = A->buffer;
    float32_t(*b)[B->shape[1]] = B->buffer;

    connx_Tensor* C = NULL;
    if (input_count == 3) {
        C = connx_Graph_get(graph, inputs[2]);
        assert(C != NULL);
    }
    if (C != NULL) {
        connx_debug("C ndim %d ", C->ndim);
        connx_Tensor_dump(C);
    }

    // Transpose A, b checks
    int A_rows;
    int A_cols;
    int B_rows;
    int B_cols;
    // A's # of cols and B's # of rowshould be same
    if (transA && transB) {
        fprintf(stderr, "Gemm: Matrix Transpose A, B\n");
        A_rows = A->shape[1];
        A_cols = A->shape[0];
        B_rows = B->shape[1];
        B_cols = B->shape[0];
    } else if (transA && !transB) {
        fprintf(stderr, "Gemm: Matrix Transpose A\n");
        A_rows = A->shape[1];
        A_cols = A->shape[0];
        B_rows = B->shape[0];
        B_cols = B->shape[1];
    } else if (!transA && transB) {
        fprintf(stderr, "Gemm: Matrix Transpose B\n");
        A_rows = A->shape[0];
        A_cols = A->shape[1];
        B_rows = B->shape[1];
        B_cols = B->shape[0];
    } else {
        fprintf(stderr, "Gemm: Matrix Transpose None\n");
        A_rows = A->shape[0];
        A_cols = A->shape[1];
        B_rows = B->shape[0];
        B_cols = B->shape[1];
    }
    fprintf(stderr, "Gemm: A_rows %d, A_cols %d, B_rows %d, B_cols %d\n", A_rows, A_cols, B_rows, B_cols);

    // check shape of A, B
    if (A_cols != B_rows) {
        connx_error("Gemm: A->shape[1] != B->shape[0]");
        return CONNX_TENSOR_SHAPE_NOT_MATCHING;
    } else {
        connx_debug("Gemm: A->shape[0] == B->shape[1]\n");
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

    if (transA)
        show_elements_transposed(A_rows, A_cols, a);
    else
        show_elements_normal(A_rows, A_cols, a);

    if (transB)
        show_elements_transposed(B_rows, B_cols, b);
    else
        show_elements_normal(B_rows, B_cols, b);

    // multiplication w/ transposed flag
    for (int32_t i = 0; i < A_rows; ++i) {     // from A row 0 to row A->shape[0]
        for (int32_t j = 0; j < B_cols; ++j) { // from B col 0 to col B->shape[1]
            float32_t sum = 0.0;
            for (int32_t k = 0; k < A_cols; ++k) { // Repeat * & sum as A # of cols == B # of rows
                float32_t a_val = transA ? a[k][i] : a[i][k];
                float32_t b_val = transB ? b[j][k] : b[k][j];
                //  float32_t a_val = a[i][k];
                //  float32_t b_val = b[k][j];
                sum += alpha * a_val * b_val; // + bC
                fprintf(stderr, "a[%d][%d] b[%d][%d]\n", i, k, k, j);
            }
            y[i][j] = sum;
        }
    }

    // bias add
    if (C != NULL) {
        float32_t(*c)[C->shape[1]] = C->buffer;
        for (int32_t i = 0; i < C->shape[0]; ++i) {
            for (int32_t j = 0; j < C->shape[1]; ++j) {
                y[i][j] += beta * c[i][j];
            }
        }
    }

    connx_debug("Y ndim %d ", Y->ndim);
    connx_Tensor_dump(Y);

    connx_Graph_set(graph, outputs[0], Y);
    return CONNX_OK;
}
