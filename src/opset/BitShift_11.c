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

// clang-format off
int BitShift_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                             // clang-format on
                             __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                             __attribute__((unused)) uint32_t attribute_count,
                             __attribute__((unused)) void** attributes) {
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* Y = connx_Graph_get(graph, inputs[1]);

    if (attribute_count < 1) {
        return CONNX_NOT_SUPPORTED_ATTRIBUTE;
    }

    if (X->dtype != Y->dtype) {
        connx_error("BitShift: X and Y must have the same dtype, not %d and %d.\n", X->dtype, Y->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    /*{% set supported_data_types = [UINT8, UINT16, UINT32, UINT64] %}*/
    switch (X->dtype) {
    /*{% for dtype in supported_data_types %}*/
    case {{ dtype }}:
        break;
    /*{% endfor %}*/
    default:
        connx_error("BitShift: Unsupported data type: %d", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    char* direction = (char*)attributes[0];
    bool is_left;
    if (strcmp(direction, "LEFT") == 0) {
        is_left = true;
    } else if (strcmp(direction, "RIGHT") == 0) {
        is_left = false;
    } else {
        connx_error("BitShift: Unsupported direction: %s", direction);
        return CONNX_NOT_SUPPORTED_ATTRIBUTE;
    }

    connx_Tensor* Z = connx_Tensor_alloc_broadcasted(X->dtype, X, Y);

    int32_t total = connx_Int32_product(Z->ndim, Z->shape);

    switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
    case {{ DTYPE }}: {
        {{TYPE}}* X_array = X->buffer;
        {{TYPE}}* Y_array = Y->buffer;
        {{TYPE}}* Z_array = Z->buffer;

        if (is_left) {
            for (int32_t i = 0; i < total; i++) {
                Z_array[i] = X_array[i] << Y_array[i];
            }
        } else {
            for (int32_t i = 0; i < total; i++) {
                Z_array[i] = X_array[i] >> Y_array[i];
            }
        }
        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("BitShift: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Z);

    return CONNX_OK;
}
