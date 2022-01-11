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
#include <string.h>

#include <connx/accel.h>
#include <connx/connx.h>

int Split(connx_Graph* graph, uint32_t output_count, uint32_t* outputs_, uint32_t input_count, uint32_t* inputs,
          void** attributes) {
    /*{% set supported_data_types = [
        INT8, INT16, INT32, INT64,
        UINT8, UINT16, UINT32, UINT64,
        FLOAT32, FLOAT64,
        BOOL,
        ] %}*/
    // TODO: STRING

    /*{% set supported_index_types = [ INT32, INT64, ] %}*/
    connx_Tensor* input = connx_Graph_get(graph, inputs[0]);
    // connx_Tensor* split = connx_Graph_get(graph, inputs[1]); // Use it later
    int64_t split[output_count];
    int32_t axis = *(int32_t*)attributes[0];

    // negative axis means counting from the end
    if (axis < 0) {
        axis += input->ndim;
    }

    assert(axis >= 0 && axis < input->ndim);

    int32_t output_ndim = input->ndim;
    int32_t output_shape[output_count][output_ndim];

    connx_Tensor* outputs[output_count];

    // Get splits if given
    if (input_count > 1) {
        connx_Tensor* split_tensor = connx_Graph_get(graph, inputs[1]);
        int64_t* split_array = (int64_t*)split_tensor->buffer;
        for (uint32_t i = 0; i < output_count; i++) {
            split[i] = split_array[i];
        }
    } else {
        // If not given, split evenly
        int64_t split_size = input->shape[axis] / output_count;
        for (uint32_t i = 0; i < output_count; i++) {
            split[i] = split_size;
        }
    }

    // Calculate output shape and create tensor
    for (uint32_t i = 0; i < output_count; i++) {
        memcpy(output_shape[i], input->shape, sizeof(int32_t) * output_ndim);
        output_shape[i][axis] = split[i];
        outputs[i] = connx_Tensor_alloc(input->dtype, output_ndim, output_shape[i]);
        if (outputs[i] == NULL) {
            // Free all previous and exit
            for (uint32_t j = 0; j < i; j++) {
                connx_free(outputs[j]);
            }
            return CONNX_NOT_ENOUGH_MEMORY;
        }
        connx_Graph_set(graph, outputs_[i], outputs[i]);
    }

    // Get data type size
    size_t data_type_size;

    switch (input->dtype) {
        /*{% for dtype in supported_data_types %}*/
    case {{ dtype }}:
        /*{% endfor %}*/
        {
            data_type_size = connx_DataType_size(input->dtype);
            break;
        }
    default:
        connx_error("Slice: Datatype %d is not supported yet.\n", input->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    int32_t total = connx_Int32_product(input->ndim, input->shape);

    // Calculate batch sizes
    int32_t batch_sizes[output_count];
    int32_t output_offsets[output_count];
    for (uint32_t i = 0; i < output_count; i++) {
        // Lower dimension size * split size
        batch_sizes[i] = connx_Int32_product(input->ndim - axis - 1, input->shape + axis + 1) * split[i];
    }
    memset(&output_offsets, 0, sizeof(int32_t) * output_count);

    for (int32_t input_offset = 0, index = 0; input_offset < total;) {
        int32_t current_batch_size = batch_sizes[index];
        memcpy((outputs[index])->buffer + output_offsets[index] * data_type_size,
               input->buffer + input_offset * data_type_size, current_batch_size * data_type_size);

        output_offsets[index] += current_batch_size;
        input_offset += current_batch_size;
        index = (index + 1) % output_count;
    }

    return CONNX_OK;
}
