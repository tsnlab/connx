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
#include <connx/accel.h>
#include <connx/connx.h>

// clang-format off
int Reshape_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                            // clang-format on
                            __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                            __attribute__((unused)) uint32_t attribute_count, void** attributes) {
    connx_Tensor* data = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* shape = connx_Graph_get(graph, inputs[1]);
    int32_t allowzero = *(int32_t*)attributes[0];

    int32_t ndim = shape->shape[0];
    int32_t new_shape[ndim];

    // Copy tensor shape to array new_shape
    int32_t negative_idx = -1;
    for (int32_t i = 0; i < ndim; i++) {
        new_shape[i] = ((int64_t*)shape->buffer)[i];

        if (allowzero == 0 && new_shape[i] == 0) {
            new_shape[i] = data->shape[i];
        }

        if (new_shape[i] == -1) {
            negative_idx = i;
        }
    }

    // Process -1 dim
    if (negative_idx >= 0) {
        int32_t total = connx_Int32_product(data->ndim, data->shape);

        new_shape[negative_idx] = 1;
        int32_t remain = connx_Int32_product(ndim, new_shape);

        new_shape[negative_idx] = total / remain;
    }

    // Make a reshaped tensor
    connx_Tensor* reshaped = connx_Tensor_reshape(data, ndim, new_shape);
    if (reshaped == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], reshaped);

    return CONNX_OK;
}
