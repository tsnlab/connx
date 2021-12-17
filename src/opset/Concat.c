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
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <connx/accel.h>
#include <connx/connx.h>

/*
   Example of concat operation:
   // 3d 2x2x2
   input[] = [
   [[ 1,  2], [ 3,  4]],
   [[ 5,  6], [ 7,  8]]
   ], [
   [[ 9, 10], [11, 12]],
   [[13, 14], [15, 16]]
   ]

   // axis = 0
   [
   [[ 1,  2], [ 3,  4]],
   [[ 5,  6], [ 7,  8]],
   [[ 9, 10], [11, 12]],
   [[13, 14], [15, 16]]
   ]

   // axis = 1
   [
   [[ 1,  2], [ 3,  4]],
   [[ 9, 10], [11, 12]],
   [[ 5,  6], [ 7,  8]],
   [[13, 14], [15, 16]]
   ]

   // axis = 2
   [
   [[ 1,  2], [ 9, 10]],
   [[ 3,  4], [11, 12]],
   [[ 5,  6], [13, 14]],
   [[ 7,  8], [15, 16]]
   ]
*/

int Concat(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
           __attribute__((unused)) uint32_t input_count, uint32_t* inputs_, __attribute__((unused)) void** attributes) {
    /* {% set SUPPORTED_DTYPES = [
       UINT8, UINT16, UINT32, UINT64,
       INT8, INT16, INT32, INT64,
       FLOAT32, FLOAT64,
       BOOL,
    ] %} */
    /*
        TODO: Should support these attributes:
        STRING,
        COMPLEX64, COMPLEX128,
    */
    int32_t axis = *(int32_t*)attributes[0];

    // Get input tensors
    connx_Tensor* inputs[input_count];
    for (uint32_t i = 0; i < input_count; i++) {
        inputs[i] = connx_Graph_get(graph, inputs_[i]);
    }

    int32_t ndim = inputs[0]->ndim;

    if (axis < 0) {
        // If axis is negative, it is counted from the end of the shape.
        axis += inputs[0]->ndim;
    }

    if (axis >= ndim) {
        // If axis is out of range, an error is returned.
        return CONNX_OUT_OF_INDEX; // FIXME: maybe not this error code
    }

    // Create output shape
    int32_t shape_overall[inputs[0]->ndim];
    memcpy(&shape_overall, inputs[0]->shape, sizeof(int32_t) * inputs[0]->ndim);
    for (uint32_t i = 1; i < input_count; i++) {
        if (inputs[i]->ndim != ndim) {
            // If the number of dimensions of any input tensor is different from the first tensor, an error is returned.
            return CONNX_TENSOR_SHAPE_NOT_MATCHING;
        }
        for (int32_t j = 0; j < ndim; j++) {
            if (j == axis) {
                shape_overall[axis] += inputs[i]->shape[axis];
            } else if (shape_overall[j] != inputs[i]->shape[j]) {
                // If the shape of any input tensor is different from the first tensor, an error is returned.
                return CONNX_TENSOR_SHAPE_NOT_MATCHING;
            }
        }
    }

    // Create output tensor
    connx_Tensor* concat_result = connx_Tensor_alloc(inputs[0]->dtype, inputs[0]->ndim, shape_overall);
    if (concat_result == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    // Do concat
    switch (inputs[0]->dtype) {
        // {% for DTYPE, TYPE in loop_types(*SUPPORTED_DTYPES) %}
    case {{ DTYPE }}: {
        {{TYPE}}* output_array = concat_result->buffer;

        if (axis == 0) { // Easist case
            // uint32_t offset = 0;
            for (uint32_t matrix_index = 0; matrix_index < input_count; matrix_index++) {
                connx_Tensor* input_tensor = inputs[matrix_index];
                {{TYPE}}* input_array = input_tensor->buffer;
                // XXX: Change this into block size and repeat until result matrix is full
                int32_t matrix_size = connx_Int32_product(input_tensor->ndim, input_tensor->shape);
                // fprintf(stderr, "input_count: %d, matrix_size: %d\n", input_count, matrix_size);

                #ifndef SLOW_CONCAT
                memcpy(  // copy the data
                    output_array,
                    input_array,
                    matrix_size * sizeof({{TYPE}})
                );
                #else
                for (int32_t i = 0; i < matrix_size; i++) {
                    output_array[i] = input_array[i];
                }
                #endif

                output_array += matrix_size;
            }

        } else {
            // TODO: implement this
            return CONNX_NOT_SUPPORTED_ATTRIBUTE;
        }

        break;
    }
        // {% endfor %}
    default:
        connx_error("Concat: Datatype %d is not supported yet.\n", inputs[0]->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], concat_result);

    return CONNX_OK;
}
