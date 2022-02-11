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
#include <stdint.h>

#include <connx/accel.h>
#include <connx/connx.h>

#define max(x, y) (((x) > (y)) ? (x) : (y))

static int32_t get_broadcasted_input_offset(const connx_Tensor* output, const connx_Tensor* input,
                                            int32_t output_offset);

int {{fname}}_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
              __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
              __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    // Inputs
    connx_Tensor* A = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* B = connx_Graph_get(graph, inputs[1]);

    assert(A != NULL && B != NULL);

    // Check input types
    if (A->dtype != B->dtype) {
        connx_error("{{fname}}: input tensors must have the same data type");
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }
    switch (A->dtype) {
        /*{% for DTYPE in supported_data_types %}*/
    case {{ DTYPE }}:
        /*{% endfor %}*/
        break;
    default:
        connx_error("{{fname}}: Unsupported data type: %d", A->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Tensor* C = connx_Tensor_alloc_broadcasted(CONNX_BOOL, A, B);

    if (C == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], C);

    int32_t total = connx_Int32_product(C->ndim, C->shape);

    // Fill outputs
    switch (A->dtype) {
        /*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
    case {{ DTYPE }}: {
        {{TYPE}}* A_array = A->buffer;
        {{TYPE}}* B_array = B->buffer;
        bool* C_array = C->buffer;

        for (int32_t i = 0; i < total; i++) {
            int32_t input_offset_a = get_broadcasted_input_offset(C, A, i);
            int32_t input_offset_b = get_broadcasted_input_offset(C, B, i);
            // clang-format off
            C_array[i] = A_array[input_offset_a] {{operator}} B_array[input_offset_b];
            // clang-format on
        }
    } break;
        /*{% endfor %}*/
    default:
        connx_error("{{fname}}: Unsupported data type: %d", A->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    return CONNX_OK;
}

static int32_t get_broadcasted_input_offset(const connx_Tensor* output, const connx_Tensor* input,
                                            int32_t output_offset) {

    int32_t* output_shape = output->shape;
    int32_t* input_shape = input->shape;
    int32_t output_idxs[output->ndim];
    int32_t input_offset = 0;

    int32_t skip_size = output->ndim - input->ndim;

    // Skip first dimensions if input has smaller dimensions
    output_shape += skip_size;
    int32_t ndim = output->ndim - skip_size;

    // Calculate output indices
    for (int32_t i = ndim - 1; i >= 0; i--) {
        output_idxs[i] = output_offset % output_shape[i];
        output_offset /= output_shape[i];
    }

    // Calculate input indices
    for (int32_t i = 0; i < input->ndim; i++) {
        input_offset *= input_shape[i];
        assert(output_shape[i] >= input_shape[i]);

        if (input_shape[i] == 1) {
            input_offset += 0;
        } else {
            input_offset += output_idxs[i];
        }
    }

    return input_offset;
}
