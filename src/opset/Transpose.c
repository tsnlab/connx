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
#include <stdio.h>

#include <connx/accel.h>
#include <connx/connx.h>

int32_t get_output_index(const int32_t ndim, const int32_t* input_shape, const int32_t* output_shape, int32_t* perm,
                         const int32_t input_index);

int Transpose(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
              __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
              __attribute__((unused)) void** attributes) {
    /* {% set supported_data_types = [
        INT8, INT16, INT32, INT64,
        UINT8, UINT16, UINT32, UINT64,
        FLOAT32, FLOAT64,
        BOOL,
        ] %}
        TODO: STRING
     */
    connx_Tensor* data = connx_Graph_get(graph, inputs[0]);

    connx_AttributeInts* perm_attr = attributes[0];

    int32_t output_ndim = data->ndim;
    int32_t output_shape[output_ndim];

    if (perm_attr->count == 0) {
        for (int32_t i = 0; i < output_ndim; i++) {
            output_shape[output_ndim - i - 1] = data->shape[i];
        }
    } else {
        for (uint32_t i = 0; i < perm_attr->count; i++) {
            output_shape[i] = data->shape[perm_attr->array[i]];
        }
    }

    connx_Tensor* transposed = connx_Tensor_alloc(data->dtype, output_ndim, output_shape);
    if (transposed == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t total = connx_Int32_product(data->ndim, data->shape);
    fprintf(stderr, "total = %d\n", total);

    switch (data->dtype) {
        // {% for DTYPE, TYPE in loop_types(FLOAT32, FLOAT64) %}
    case {{ DTYPE }}: {
        {{TYPE}}* data_array = data->buffer;
        {{TYPE}}* output_array = transposed->buffer;

        for (int32_t input_index = 0; input_index < total; input_index++) {
            int32_t output_index =
                get_output_index(data->ndim, data->shape, transposed->shape, perm_attr->array, input_index);
            output_array[output_index] = data_array[input_index];
        }
        break;
    }
        // {% endfor %}
    default:
        connx_error("Transpose: Datatype %d is not supported yet.\n", data->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], transposed);

    return CONNX_OK;
}

int32_t get_output_index(const int32_t ndim, const int32_t* input_shape, const int32_t* output_shape, int32_t* perm,
                         const int32_t input_index) {
    int32_t remaining = input_index;
    int32_t output_index = 0;

    int32_t input_indexes[ndim];
    int32_t output_indexes[ndim];

    for (int32_t i = 0; i < ndim; i++) {
        input_indexes[ndim - i - 1] = remaining % input_shape[ndim - i - 1];
        remaining /= input_shape[ndim - i - 1];
    }

    for (int32_t i = 0; i < ndim; i++) {
        output_indexes[i] = input_indexes[perm[i]];
    }

    for (int32_t i = 0; i < ndim; i++) {
        output_index *= output_shape[i];
        output_index += output_indexes[i];
    }

    return output_index;
}