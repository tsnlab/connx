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
#include <connx/accel.h>
#include <connx/connx.h>

TEMPLATE_START(FLOAT32, FLOAT64, UINT32, UINT64, INT32, INT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
#define connx_TEMPLATE_NAME_broadcast connx_Float32_broadcast
static TEMPLATE_TYPE* get_TEMPLATE_NAME_row(int32_t temp_count, TEMPLATE_TYPE* temp, int32_t array_count,
                                            TEMPLATE_TYPE* array, int32_t row) {
    if (array_count >= temp_count) {
        return array + row * array_count;
    } else {
        connx_TEMPLATE_NAME_broadcast(temp_count, temp, array_count, array + row * array_count);
        return temp;
    }
}

static TEMPLATE_TYPE* get_TEMPLATE_NAME_col(int32_t temp_count, TEMPLATE_TYPE* temp, int32_t array_count,
                                            TEMPLATE_TYPE* array, int32_t col) {
    for (int32_t i = 0; i < temp_count; i++) {
        temp[i] = array[i * array_count + col];
    }
    return temp;
}
TEMPLATE_END()

int MatMul_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
           __attribute__((unused)) uint32_t input_count, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    connx_Tensor* A = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* B = connx_Graph_get(graph, inputs[1]);

    // Create Y
    int32_t ndim = A->ndim > B->ndim ? A->ndim : B->ndim;
    int32_t shape[ndim];
    for (int32_t i = 0; i < ndim; i++) { // back to front
        int32_t A_dim = i < A->ndim ? A->shape[A->ndim - 1 - i] : 0;
        int32_t B_dim = i < B->ndim ? B->shape[B->ndim - 1 - i] : 0;

        if (i == 0) {
            A_dim = 0;
        } else if (i == 1) {
            B_dim = 0;
        }

        shape[ndim - i - 1] = A_dim > B_dim ? A_dim : B_dim;
    }

    connx_Tensor* Y = connx_Tensor_alloc(A->dtype, ndim, shape);
    if (Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t A_row = A->shape[A->ndim - 2];                    // row count
    int32_t A_col = A->shape[A->ndim - 1];                    // col count
    int32_t A_unit = A_row * A_col;                           // size of matrix
    int32_t A_total = connx_Int32_product(A->ndim, A->shape); // Number of matrics
    int32_t B_row = B->shape[B->ndim - 2];
    int32_t B_col = B->shape[B->ndim - 1];
    int32_t B_unit = B_row * B_col;
    int32_t B_total = connx_Int32_product(B->ndim, B->shape);
    int32_t Y_row = shape[ndim - 2];
    int32_t Y_col = shape[ndim - 1];
    int32_t Y_unit = Y_row * Y_col;
    int32_t Y_total = connx_Int32_product(Y->ndim, Y->shape);

    switch (A->dtype) {
        TEMPLATE_START(FLOAT32, FLOAT64, UINT32, UINT64, INT32, INT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
#define connx_TEMPLATE_NAME_mul connx_Float32_mul
#define connx_TEMPLATE_NAME_sum connx_Float32_sum
    case TEMPLATE_DTYPE: {
        TEMPLATE_TYPE* A_array = A->buffer;
        TEMPLATE_TYPE* B_array = B->buffer;
        TEMPLATE_TYPE* Y_array = Y->buffer;

        int32_t count = A_col > B_row ? A_col : B_row; // count = MAX(A_col, B_row)
        TEMPLATE_TYPE tmp_a[count];
        TEMPLATE_TYPE tmp_b[count];
        TEMPLATE_TYPE tmp_mul[count];

        for (int32_t Y_idx = 0, A_idx = 0, B_idx = 0; Y_idx < Y_total;
             Y_idx += Y_unit, A_idx = (A_idx + A_unit) % A_total, B_idx = (B_idx + B_unit) % B_total) {

            for (int32_t col_idx = 0; col_idx < Y_col; col_idx++) {
                TEMPLATE_TYPE* b = get_TEMPLATE_NAME_col(count, tmp_a, B_col, B_array + B_idx, col_idx);

                for (int32_t row_idx = 0; row_idx < Y_row; row_idx++) {
                    TEMPLATE_TYPE* a = get_TEMPLATE_NAME_row(count, tmp_b, A_col, A_array + A_idx, row_idx);

                    // Mul
                    connx_TEMPLATE_NAME_mul(count, tmp_mul, a, b);

                    // Sum
                    Y_array[Y_idx + row_idx * Y_col + col_idx] = connx_TEMPLATE_NAME_sum(count, tmp_mul);
                }
            }
        }
        break;
    }
        TEMPLATE_END()
    default:
        connx_error("MatMul: Datatype %d is not supported yet.\n", A->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
