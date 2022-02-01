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
#include <assert.h>

#include <connx/accel.h>
#include <connx/connx.h>

/*{% set supported_dtypes = ['FLOAT32', 'FLOAT64'] %}*/

int GlobalMaxPool_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                  __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                  __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    // input
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);

    int32_t batch_count = X->shape[0];
    int32_t channel_count = X->shape[1];
    int32_t unit = connx_Int32_product(X->ndim - 2, X->shape + 2);
    // int32_t unit = 1;
    // for (int32_t i = 2; i < X->ndim; i++) {
    //     unit *= X->shape[i];
    // }

    assert(unit != 0);

    // output
    int32_t Y_shape[X->ndim];
    Y_shape[0] = X->shape[0];
    Y_shape[1] = X->shape[1];
    for (int i = 2; i < X->ndim; i++) {
        Y_shape[i] = 1;
    }
    connx_Tensor* Y = connx_Tensor_alloc(X->dtype, X->ndim, Y_shape);

    switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(*supported_dtypes) %}*/
    case {{ DTYPE }}: {
        {{TYPE}}* X_array = X->buffer;
        {{TYPE}}* Y_array = Y->buffer;

        for (int32_t batch = 0; batch < batch_count; batch++) {
            for (int32_t channel = 0; channel < channel_count; channel++) {
                // XXX: Assume that there is no dimension with size 0.
                // clang-format off
                {{TYPE}} max = *X_array;
                // clang-format on
                X_array += 1;
                for (int32_t i = 1; i < unit; i++) {
                    if (*X_array > max) {
                        max = *X_array;
                    }
                    X_array += 1;
                }

                *Y_array = max;
                Y_array += 1;
            }
        }
        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("GlobalMaxPool: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
