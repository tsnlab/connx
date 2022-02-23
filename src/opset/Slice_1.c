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
#include <string.h>

#include <connx/accel.h>
#include <connx/connx.h>

static int32_t get_input_index(const int32_t ndim, const int32_t* input_shape, const int32_t* output_shape,
                               const int32_t* starts, const int32_t output_offset);

// clang-format off
int Slice_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                          // clang-format on
                          __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                          __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    /*{% set supported_data_types = [
        INT8, INT16, INT32, INT64,
        UINT8, UINT16, UINT32, UINT64,
        FLOAT32, FLOAT64,
        BOOL,
        ] %}*/
    // TODO: STRING

    /*{% set supported_index_types = [ INT32, INT64, ] %}*/
    connx_Tensor* data = connx_Graph_get(graph, inputs[0]);

    connx_AttributeInts* attr_axes = attributes[0];
    connx_AttributeInts* attr_ends = attributes[1];
    connx_AttributeInts* attr_starts = attributes[2];

    int32_t output_ndim = attr_starts->count;
    int32_t output_shape[output_ndim];

    int32_t axes[output_ndim];
    int32_t ends[output_ndim];
    int32_t starts[output_ndim];

    // fill axes
    for (int32_t i = 0; i < output_ndim; i++) {
        axes[i] = attr_axes != NULL ? attr_axes->array[i] : i;
        if (axes[i] < 0) {
            axes[i] += output_ndim;
        }
    }

    // starts, ends
    for (int32_t i = 0; i < output_ndim; i++) {
        starts[i] = attr_starts->array[i];
        if (starts[i] < 0) {
            starts[i] += data->shape[axes[i]];
        } else if (starts[i] > data->shape[i]) {
            starts[i] = data->shape[i];
        }

        ends[i] = attr_ends->array[i];
        if (ends[i] < 0) {
            ends[i] += data->shape[axes[i]];
        } else if (ends[i] > data->shape[i]) {
            ends[i] = data->shape[i];
        }

        output_shape[axes[i]] = ends[i] - starts[i];
    }

    // output tensor
    connx_Tensor* output = connx_Tensor_alloc(data->dtype, output_ndim, output_shape);
    if (output == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], output);

    // Get data type size
    size_t data_type_size = connx_DataType_size(data->dtype);
    ;
    int32_t total = connx_Int32_product(output_ndim, output_shape);

    // Somethimes, one of axis is 0, which means total is 0
    // If without this check, the following loop will be endless
    if (total == 0) {
        return CONNX_OK;
    }

    int64_t batch_size = 1;
    for (int32_t i = output_ndim - 1; i >= 0; i--) {
        batch_size *= output_shape[i];

        if (starts[i] != 0 || ends[i] != output_shape[i]) {
            break;
        }
    }

    for (int64_t output_offset = 0; output_offset < total; output_offset += batch_size) {
        int64_t input_offset = get_input_index(output_ndim, data->shape, output_shape, starts, output_offset);
        memcpy(output->buffer + output_offset * data_type_size, data->buffer + input_offset * data_type_size,
               batch_size * data_type_size);
    }

    return CONNX_OK;
}

int32_t get_input_index(const int32_t ndim, const int32_t* input_shape, const int32_t* output_shape,
                        const int32_t* starts, const int32_t output_offset) {
    int32_t remaining = output_offset;
    int32_t input_offset = 0;

    int32_t input_indexes[ndim];
    int32_t output_indexes[ndim];

    for (int32_t i = 0; i < ndim; i++) {
        output_indexes[ndim - i - 1] = remaining % output_shape[ndim - i - 1];
        remaining /= output_shape[ndim - i - 1];
    }

    for (int32_t i = 0; i < ndim; i++) {
        input_indexes[i] = starts[i] + output_indexes[i];
    }

    for (int32_t i = 0; i < ndim; i++) {
        input_offset *= input_shape[i];
        input_offset += input_indexes[i];
    }

    return input_offset;
}
