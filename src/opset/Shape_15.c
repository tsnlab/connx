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

int Shape_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
          __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
          __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    connx_Tensor* data = connx_Graph_get(graph, inputs[0]);

    // end is optional, start is optional with default value from onnx-connx

    int32_t end = (attributes[0] == NULL) ? data->ndim : *(int32_t*)attributes[0];
    int32_t start = *(int32_t*)attributes[1];

    // handle negative start and end
    if (end < 0) {
        end += data->ndim;
    }

    if (start < 0) {
        start += data->ndim;
    }

    // clip start/end to 0 .. data->ndim
    start = (start < 0) ? 0 : (start > data->ndim) ? data->ndim : start;
    end = (end < 0) ? 0 : (end > data->ndim) ? data->ndim : end;

    int32_t output_shape[1] = {end - start};

    connx_Tensor* shape = connx_Tensor_alloc(INT64, 1, output_shape);
    if (shape == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int64_t* array = shape->buffer;

    for (int i = 0; i < (end - start); i++) {
        array[i] = data->shape[start + i];
    }

    connx_Graph_set(graph, outputs[0], shape);

    return CONNX_OK;
}
