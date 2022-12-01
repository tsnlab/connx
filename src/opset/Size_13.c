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

#include <connx/accel.h>
#include <connx/connx.h>

// clang-format off
int Size_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                         // clang-format on
                         __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                         __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    connx_Tensor* data = connx_Graph_get(graph, inputs[0]);
    // clang-format off
    connx_Tensor* size = connx_Tensor_alloc(INT64, 1, (int32_t[]){1,});
    // clang-format on
    if (size == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t total = connx_Int32_product(data->ndim, data->shape);
    int64_t* size_buffer = (int64_t*)size->buffer;

    size_buffer[0] = total;

    connx_Graph_set(graph, outputs[0], size);

    return CONNX_OK;
}
