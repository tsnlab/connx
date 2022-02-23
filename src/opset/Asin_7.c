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
int Asin_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                         // clang-format on
                         __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                         __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    connx_Tensor* input = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* output = connx_Tensor_alloc_like(input);

    int32_t total = connx_Int32_product(input->ndim, input->shape);

    switch (input->dtype) {
        /*{% for DTYPE, TYPE in loop_types(FLOAT32, FLOAT64) %}*/
    case {{ DTYPE }}: {
        {{TYPE}}* input_array = input->buffer;
        {{TYPE}}* output_array = output->buffer;

        for (int32_t i = 0; i < total; i++) {
            output_array[i] = asinf(input_array[i]);
        }
        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("Asin: Datatype %d is not supported yet.\n", input->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], output);

    return CONNX_OK;
}
