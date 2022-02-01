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

/*{% set supported_dtypes = [
    UINT8, UINT16, UINT32, UINT64,
    INT8, INT16, INT32, INT64,
    FLOAT32, FLOAT64,
    BOOL,
]
%}*/
// TODO: STRING,

static inline void get_indices(int32_t ndim, int32_t* shape, int32_t offset, int32_t* indices) {
    for (int32_t i = ndim - 1; i >= 0; i--) {
        indices[i] = offset % shape[i];
        offset /= shape[i];
    }
}

int NonZero_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
            __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
            __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);

    int32_t total = connx_Int32_product(X->ndim, X->shape);

    // First, Count all nonzero elements
    int32_t nonzero_count = 0;

    switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(*supported_dtypes) %}*/
    case {{ DTYPE }}: {
        {{TYPE}}* X_array = X->buffer;
        for (int32_t i = 0; i < total; i++) {
            nonzero_count += X_array[i] != 0 ? 1 : 0;
        }
        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("NonZero: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    // Now, create output and fill it with nonzero elements' indices
    int32_t output_ndim = 2;
    int32_t output_shape[2] = {X->ndim, nonzero_count};
    connx_Tensor* Y = connx_Tensor_alloc(INT64, output_ndim, output_shape);
    if (Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(*supported_dtypes) %}*/
    case {{ DTYPE }}: {
        {{TYPE}}* X_array = X->buffer;
        int64_t* Y_array = Y->buffer;
        int32_t output_offset = 0;
        int32_t indices[X->ndim];

        // Don't need to loop more if output is full.
        // So use output_offset < nonzero_count instead of i < total.
        for (int32_t i = 0; output_offset < nonzero_count; i++) {
            if (X_array[i] != 0) {
                get_indices(X->ndim, X->shape, i, indices);
                for (int32_t j = 0; j < X->ndim; j++) {
                    Y_array[j * nonzero_count + output_offset] = indices[j];
                }
                output_offset++;
            }
        }
    } break;
        /*{% endfor %}*/
    default:
        connx_error("NonZero: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);
    return CONNX_OK;
}
