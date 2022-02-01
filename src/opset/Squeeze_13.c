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
#include <stdint.h>
#include <string.h>

#include <connx/accel.h>
#include <connx/connx.h>

int Squeeze_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
        uint32_t input_count, uint32_t* inputs,
        __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    // Inputs
    const connx_Tensor* data = connx_Graph_get(graph, inputs[0]);
    const connx_Tensor* axes = input_count > 1 ? connx_Graph_get(graph, inputs[1]) : NULL;

    if (axes != NULL) {
        assert(axes->ndim == 1);
        assert(axes->dtype == INT64);
    }

    // Doesn't matter of data type. It just removes the axes with 1.
    // Actual buffer data remains still.

    int32_t output_ndim = data->ndim;
    int32_t output_shape[output_ndim];

    // Check and adjust axes
    if (axes != NULL) {
        int64_t axes_array[axes->shape[0]];
        memcpy(axes_array, axes->buffer, axes->size);

        for (int32_t i = 0; i < axes->shape[0]; i++) {
            if (axes_array[i] < 0) {
                axes_array[i] += data->ndim;
            }
            if (axes_array[i] < 0 || axes_array[i] >= data->ndim || data->shape[axes_array[i]] != 1) {
                connx_error("Squeeze: Invalid axes. %d is not a valid axis for %d-D tensor.", axes_array[i],
                            data->ndim);
                return CONNX_TENSOR_SHAPE_NOT_MATCHING;
            }
        }
        // Now fill the output shape
        memcpy(output_shape, data->shape, data->ndim * sizeof(output_shape[0]));
        for (int32_t i = 0; i < axes->shape[0]; i++) {
            int32_t* dst = output_shape + axes_array[i];
            int32_t* src = output_shape + axes_array[i] + 1;
            size_t amount = (output_ndim - axes_array[i] - 1);

            // memcpy claims that overlap. Use memmove
            memmove(dst, src, amount * sizeof(output_shape[0]));
            output_ndim -= 1;
        }
    } else {
        // Remove all axis with 1
        output_ndim = 0;
        for (int32_t i = 0; i < data->ndim; i++) {
            if (data->shape[i] != 1) {
                output_shape[output_ndim] = data->shape[i];
                output_ndim += 1;
            }
        }
    }

    connx_Tensor* squeezed = connx_Tensor_alloc(data->dtype, output_ndim, output_shape);

    if (squeezed == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], squeezed);

    // Fill outputs
    memcpy(squeezed->buffer, data->buffer, data->size);

    return CONNX_OK;
}
