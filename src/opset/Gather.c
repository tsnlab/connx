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

int Gather(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
           __attribute__((unused)) uint32_t input_count, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    /*
        {% set supported_dtypes = [
            UINT8,
            UINT16,
            UINT32,
            UINT64,
            INT8,
            INT16,
            INT32,
            INT64,
            FLOAT32,
            FLOAT64,
            BOOL,
        ]
        %}
        TODO:
        STRING,
    */
    // inputs
    connx_Tensor* data = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* indices = connx_Graph_get(graph, inputs[1]);

    // attributes
    int32_t axis = *(int32_t*)attributes[0];

    int32_t ndim_data = data->ndim;
    int32_t ndim_indices = indices->ndim;

    if (axis < 0) {
        // If axis is negative, it is counted from the end of the shape.
        axis += ndim_data;
    }

    if (axis >= ndim_data) {
        // If axis is out of range, an error is returned.
        return CONNX_OUT_OF_INDEX;
    }

    // Calculate output shape
    int32_t ndim_output = ndim_data - 1 + ndim_indices;

    // shape_output = ndim_data[0:axis] + ndim_indices + ndim_data[axis+1:]
    int32_t shape_output[ndim_output];

    // use memcpy
    memcpy(shape_output, data->shape, sizeof(int32_t) * axis);
    memcpy(shape_output + axis, indices->shape, sizeof(int32_t) * ndim_indices);
    memcpy(shape_output + axis + ndim_indices, data->shape + axis + 1, sizeof(int32_t) * (ndim_data - axis - 1));

    // create output
    connx_Tensor* output = connx_Tensor_alloc(data->dtype, ndim_output, shape_output);

    if (output == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    size_t datatype_size;

    // get datatype size
    switch (data->dtype) {
        // {% for DTYPE, TYPE in loop_types(*supported_dtypes) %}
    case {{ DTYPE }}: {
        datatype_size = sizeof({{TYPE}});
        break;
    }
        // {% endfor %}
    default:
        connx_error("Concat: Datatype %d is not supported yet.\n", data->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    // do gather
    switch (indices->dtype) {
        // {% for DTYPE, TYPE in loop_types(INT32, INT64) %}
    case {{ DTYPE }}: {
        // outer = data[0:axis]
        int32_t outer_loop_count = connx_Int32_product(axis, data->shape);
        // inner = indices[:]
        int32_t inner_loop_count = connx_Int32_product(indices->ndim, indices->shape);
        // block = data[axis+1:]
        int32_t outer_block_count = connx_Int32_product(data->ndim - axis, data->shape + axis);
        int32_t inner_block_count = connx_Int32_product(data->ndim - axis - 1, data->shape + axis + 1);

        // check indices boundary first
        int32_t total_indices = connx_Int32_product(indices->ndim, indices->shape);
        {{TYPE}}* indices_array = indices->buffer;
        for (int32_t i = 0; i < total_indices; i++) {
            if (indices_array[i] >= data->shape[axis]) {
                connx_error("Gather: indices[%d] is out of data shape[%d].\n", indices_array[i], data->shape[axis]);
                // free output
                connx_free(output);
                return CONNX_OUT_OF_INDEX;
            }
        }

        for (int32_t outer_index = 0; outer_index < outer_loop_count; outer_index += 1) {
            for (int32_t inner_index = 0; inner_index < inner_loop_count; inner_index += 1) {

                // Get index
                int32_t index = indices_array[inner_index];
                if (index < 0) {
                    index += data->shape[axis];
                }

                // Hang tight!
                int32_t output_data_index = (outer_index * inner_loop_count + inner_index) * inner_block_count;
                int32_t input_data_index = (outer_index * outer_block_count) + (index * inner_block_count);

                // Calculate pointer
                void* output_ptr = output->buffer + output_data_index * datatype_size;
                void* data_ptr = data->buffer + input_data_index * datatype_size;

                // Copy data
                memcpy(output_ptr, data_ptr, inner_block_count * datatype_size);
            }
        }

        break;
    }
        // {% endfor %}
    default:
        connx_error("Gather: Datatype %d is not supported yet.\n", indices->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], output);

    return CONNX_OK;
}
