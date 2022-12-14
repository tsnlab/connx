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
#include <assert.h>
#include <math.h>
#include <stdint.h>

#include <connx/accel.h>
#include <connx/connx.h>

// clang-format off
int Pow_{{ op_version }}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                        // clang-format on
                        __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                        __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    // Inputs
    connx_Tensor* A = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* B = connx_Graph_get(graph, inputs[1]);

    assert(A != NULL && B != NULL);

    /*{% set supported_A_types = (INT32, INT64, FLOAT32, FLOAT64) %}*/
    /*{% set supported_B_types = (
             UINT8, UINT16, UINT32, UINT64,
             INT8, INT16, INT32, INT64,
             FLOAT32, FLOAT64) %}*/

    // Check input types
    switch (A->dtype) {
        /*{% for DTYPE in supported_A_types %}*/
    case {{ DTYPE }}:
        /*{% endfor %}*/
        break;
    default:
        connx_error("Pow: Unsupported data type A: %d", A->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }
    switch (B->dtype) {
        /*{% for DTYPE in supported_B_types %}*/
    case {{ DTYPE }}:
        /*{% endfor %}*/
        break;
    default:
        connx_error("Pow: Unsupported data type B: %d", A->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Tensor* C = connx_Tensor_alloc_broadcasted(A->dtype, A, B);

    if (C == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], C);

    int32_t total = connx_Int32_product(C->ndim, C->shape);

    // Fill outputs
    switch (A->dtype) {
        /*{% for DTYPE_A, TYPE_A in loop_types(*supported_A_types) %}*/
    case {{ DTYPE_A }}: {
        switch (B->dtype) {
            /*{% for DTYPE_B, TYPE_B in loop_types(*supported_B_types) %}*/
        case {{ DTYPE_B }}: {
            {{TYPE_A}}* A_array = A->buffer;
            {{TYPE_B}}* B_array = B->buffer;
            {{TYPE_A}}* C_array = C->buffer;

            /*{% set func = 'powf' if DTYPE_A == FLOAT32 else 'pow' %}*/

            bool is_likely = connx_Tensor_is_likely_same_shape(A, B);

            for (int32_t i = 0; i < total; i++) {
                int32_t input_offset_a = is_likely ? i : connx_Tensor_get_broadcasted_input_offset(C, A, i);
                int32_t input_offset_b = is_likely ? i : connx_Tensor_get_broadcasted_input_offset(C, B, i);
                C_array[i] = {{func}}(A_array[input_offset_a], B_array[input_offset_b]);
            }
        } break;
            /*{% endfor %}*/
        default:
            connx_error("Pow: Unsupported data type: %d", A->dtype);
            return CONNX_NOT_SUPPORTED_DATATYPE;
        }
    } break;
    /*{% endfor %}*/
    default:
        connx_error("Pow: Unsupported data type: %d", A->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    return CONNX_OK;
}
