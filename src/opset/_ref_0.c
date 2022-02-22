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
#include <connx/connx.h>

int _ref_0(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, __attribute__((unused)) uint32_t* outputs,
           __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
           __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {

    connx_Tensor* tensor = connx_Graph_get(graph, inputs[0]);
    int32_t ref_count = *(int32_t*)attributes[0];

    connx_Lock_lock(&tensor->lock);
    tensor->ref_count += ref_count;
    connx_Lock_unlock(&tensor->lock);

    return CONNX_OK;
}
