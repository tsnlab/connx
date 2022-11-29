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
int Mod_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                        // clang-format on
                        __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                        __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    connx_Tensor* A = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* B = connx_Graph_get(graph, inputs[1]);
    connx_Tensor* C = connx_Tensor_alloc_broadcasted(A->dtype, A, B);
    if (C == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    bool is_fmod = false;
    if (attribute_count > 0) {
        is_fmod = *(bool*)attributes[0];
    }

    int32_t total = connx_Int32_product(C->ndim, C->shape);
    bool is_likely_shape = connx_Tensor_is_likely_same_shape(A, B);

    switch (A->dtype) {
        /*{% for DTYPE, TYPE in loop_types(
                 UINT8, UINT16, UINT32, UINT64,
                 INT8, INT16, INT32, INT64,
                 FLOAT32, FLOAT64) %}*/
    case {{ DTYPE }}: {
        {{TYPE}}* A_array = A->buffer;
        {{TYPE}}* B_array = B->buffer;
        {{TYPE}}* C_array = C->buffer;

        /*{% if DTYPE in (FLOAT32, FLOAT64) %}*/
        for (int32_t i = 0; i < total; i++) {
            int32_t input_offset_a = is_likely_shape ? i : connx_Tensor_get_broadcasted_input_offset(C, A, i);
            int32_t input_offset_b = is_likely_shape ? i : connx_Tensor_get_broadcasted_input_offset(C, B, i);
            C_array[i] = fmod(A_array[input_offset_a], B_array[input_offset_b]);
        }
        /*{% else %}*/
        if (is_fmod) {
            for (int32_t i = 0; i < total; i++) {
                int32_t input_offset_a = is_likely_shape ? i : connx_Tensor_get_broadcasted_input_offset(C, A, i);
                int32_t input_offset_b = is_likely_shape ? i : connx_Tensor_get_broadcasted_input_offset(C, B, i);
                C_array[i] = fmod(A_array[input_offset_a], B_array[input_offset_b]);
            }
        } else {
            for (int32_t i = 0; i < total; i++) {
                int32_t input_offset_a = is_likely_shape ? i : connx_Tensor_get_broadcasted_input_offset(C, A, i);
                int32_t input_offset_b = is_likely_shape ? i : connx_Tensor_get_broadcasted_input_offset(C, B, i);
                C_array[i] = A_array[input_offset_a] % B_array[input_offset_b];
                /*{% if DTYPE in (INT8, INT16, INT32, INT64) %}*/
                // The sign of the remainder is the same as that of the Divisor.
                if (B_array[input_offset_b] * C_array[i] < 0) {
                    C_array[i] += B_array[input_offset_b];
                }
                /*{% endif %}*/
            }
        }
        /*{% endif %}*/
        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("Mod: Datatype %d is not supported yet.\n", A->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], C);

    return CONNX_OK;
}
