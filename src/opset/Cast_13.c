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
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <connx/accel.h>
#include <connx/connx.h>

// clang-format off
int Cast_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                         // clang-format on
                         __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                         __attribute__((unused)) uint32_t attribute_count, void** attributes) {
    connx_DataType to = *(int32_t*)attributes[0];
    connx_Tensor* input = connx_Graph_get(graph, inputs[0]);
    // FIXME: Remove this block after support string type
    if (to == CONNX_STRING || input->dtype == CONNX_STRING) {
        connx_error("Cast: Datatype %d is not supported yet.\n", to);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Tensor* output = connx_Tensor_alloc(to, input->ndim, input->shape);
    if (output == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t total = connx_Int32_product(input->ndim, input->shape);

    switch (to) {
    /*{% for DTYPE_output, TYPE_output in loop_types(FLOAT32, FLOAT64, INT8, INT16, INT32, INT64, UINT8, UINT16,
     UINT32, UINT64, BOOL, STRING) %}*/
    case {{ DTYPE_output }}: {
        {{TYPE_output}}* output_array = output->buffer;

        switch (input->dtype) {
            /*{% for DTYPE_input, TYPE_input in loop_types(FLOAT32, FLOAT64, INT8, INT16, INT32, INT64, UINT8, UINT16,
             UINT32, UINT64, BOOL, STRING) %}*/
        case {{ DTYPE_input }}: {
            {{TYPE_input}}* input_array = input->buffer;

            for (int32_t i = 0; i < total; i++) {
                /*{% if DTYPE_output == STRING and DTYPE_input == STRING %}*/
                strcpy(output_array[i], input_array[i]); // FIXME: strncpy?
                /*{% elif DTYPE_output == STRING %}*/
                /*{%   if DTYPE_input in (FLOAT32, FLOAT64) %}*/
                sprintf(output_array[i], "%g", input_array[i]); // FIXME: float to string
                /*{%   elif DTYPE_input in (INT64, UINT64) %}*/
                sprintf(output_array[i], "%" PRId64, input_array[i]);
                /*{%   else %}*/
                sprintf(output_array[i], "%d", input_array[i]);
                /*{%   endif %}*/
                /*{% elif DTYPE_input == STRING %}*/
                /*{%   if DTYPE_output in (FLOAT32, FLOAT64) %}*/
                output_array[i] = atof(input_array[i]); // FIXME: string to float
                /*{%   else %}*/
                output_array[i] = atoi(input_array[i]);
                /*{%   endif %}*/
                /*{% else %}*/
                output_array[i] = input_array[i];
                /*{% endif %}*/
            }
            break;
        }
            /*{% endfor %}*/ // DTYPE_input, TYPE_input
        default:
            connx_error("Cast: Datatype %d on input is not supported yet.\n", input->dtype);
            return CONNX_NOT_SUPPORTED_DATATYPE;
        }

        break;
    }
        /*{% endfor %}*/ // DTYPE_output, TYPE_output
    default:
        connx_error("Cast: Datatype %d on output is not supported yet.\n", output->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], output);

    return CONNX_OK;
}
