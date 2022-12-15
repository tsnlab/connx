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
#include <stdlib.h>

#include <connx/accel.h>
#include <connx/connx.h>

// clang-format off
int Bernoulli_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                              // clang-format on
                              __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                              __attribute__((unused)) uint32_t attribute_count,
                              __attribute__((unused)) void** attributes) {
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);

    int32_t dtype = X->dtype;
    if (attribute_count > 0) {
        dtype = *(int32_t*)attributes[0];
    }

    float32_t seed = 0.0;
    if (attribute_count > 1) {
        seed = *(float32_t*)attributes[1];
    } else {
        seed = 0;  // XXX: Randomly picked
    }

    /*{% set supported_data_types = [FLOAT32, FLOAT64] %}*/
    switch (X->dtype) {
    /*{% for dtype in supported_data_types %}*/
    case {{ dtype }}:
        break;
    /*{% endfor %}*/
    default:
        connx_error("Bernoulli: Unsupported data type: %d", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    /*{% set supported_output_data_types = [
             FLOAT32, FLOAT64,
             UINT8, UINT16, UINT32, UINT64,
             INT8, INT16, INT32, INT64,
             BOOL,
         ] %}*/
    switch (dtype) {
    /*{% for dtype in supported_output_data_types %}*/
    case {{ dtype }}:
    /*{% endfor %}*/
        break;
    default:
        connx_error("Bernoulli: Unsupported output data type: %d", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Tensor* Y = connx_Tensor_alloc(dtype, X->ndim, X->shape);

    int32_t total = connx_Int32_product(Y->ndim, Y->shape);

    srand(seed);

    switch (X->dtype) {
        /*{% for DTYPE_X, TYPE_X in loop_types(*supported_data_types) %}*/
    case {{ DTYPE_X }}: {
        switch (dtype) {
            /*{% for DTYPE_Y, TYPE_Y in loop_types(*supported_output_data_types) %}*/
        case {{ DTYPE_Y }}: {
            {{TYPE_X}}* X_array = X->buffer;
            {{TYPE_Y}}* Y_array = Y->buffer;

            for (int32_t i = 0; i < total; i++) {
                Y_array[i] = X_array[i] > (float32_t)(rand() / (RAND_MAX + 1.0));
            }

            break;
        }
            /*{% endfor %}*/
        default:
            connx_error("Bernoulli: Datatype %d is not supported yet.\n", dtype);
            return CONNX_NOT_SUPPORTED_DATATYPE;
            }
        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("Bernoulli: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
