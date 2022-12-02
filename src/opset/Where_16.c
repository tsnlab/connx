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
#include <math.h>

#include <connx/accel.h>
#include <connx/connx.h>

// clang-format off
int Where_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                          // clang-format on
                          __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                          __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    connx_Tensor* condition = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* X = connx_Graph_get(graph, inputs[1]);
    connx_Tensor* Y = connx_Graph_get(graph, inputs[2]);

    /*{% set supported_data_types = [
             UINT8, UINT16, UINT32, UINT64,
             INT8, INT16, INT32, INT64,
             FLOAT32, FLOAT64,
             STRING, BOOL,
         ] %}*/

    // Check types
    if (condition->dtype != BOOL) {
        connx_error("Where: condition's dtype must be boolean, not %d.\n", condition->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    if (X->dtype != Y->dtype) {
        connx_error("Where: X and Y must have the same dtype, not %d and %d.\n", X->dtype, Y->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    switch (X->dtype) {
    /*{% for dtype in supported_data_types %}*/
    case {{ dtype }}:
        break;
    /*{% endfor %}*/
    default:
        connx_error("Where: Unsupported data type: %d", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Tensor* output = connx_Tensor_alloc_broadcasted(X->dtype, X, Y);
    if (output == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t total = connx_Int32_product(output->ndim, output->shape);

    switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
    case {{ DTYPE }}: {
        bool* condition_array = condition->buffer;
        {{TYPE}}* X_array = X->buffer;
        {{TYPE}}* Y_array = Y->buffer;
        {{TYPE}}* output_array = output->buffer;

        for (int32_t i = 0; i < total; i++) {
            int32_t input_offset_x = connx_Tensor_get_broadcasted_input_offset(output, X, i);
            int32_t input_offset_y = connx_Tensor_get_broadcasted_input_offset(output, Y, i);
            // TODO: handle string using memcpy or strcpy
            if (condition_array[i]) {
                output_array[i] = X_array[input_offset_x];
            } else {
                output_array[i] = Y_array[input_offset_y];
            }
        }
        break;
    }
        /*{% endfor %}*/
    }

    connx_Graph_set(graph, outputs[0], output);

    return CONNX_OK;
}
