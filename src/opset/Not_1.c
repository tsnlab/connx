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
#include <assert.h>
#include <stdint.h>

#include <connx/accel.h>
#include <connx/connx.h>

int Not_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
        __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
        __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    // Inputs
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);

    // Check input types
    if (X->dtype != BOOL) {
        connx_error("Not: Unsupported data type: %d", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Tensor* Y = connx_Tensor_alloc_like(X);

    if (Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], Y);

    int32_t total = connx_Int32_product(Y->ndim, Y->shape);

    // Fill outputs
    bool* X_array = X->buffer;
    bool* Y_array = Y->buffer;

    for (int32_t i = 0; i < total; i++) {
        Y_array[i] = !X_array[i];
    }

    return CONNX_OK;
}
