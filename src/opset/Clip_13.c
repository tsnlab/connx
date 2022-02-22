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

#include <connx/accel.h>
#include <connx/connx.h>

// clang-format off
int Clip_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                         // clang-format on
                         uint32_t input_count, uint32_t* inputs, __attribute__((unused)) uint32_t attribute_count,
                         __attribute__((unused)) void** attributes) {
    // input
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* Y = connx_Tensor_alloc(X->dtype, X->ndim, X->shape);

    switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(FLOAT16, FLOAT32, FLOAT64, UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64) %}*/
    case {{ DTYPE }}: {
        {{TYPE}} min = CONNX_{{ DTYPE }}_MIN;
        {{TYPE}} max = CONNX_{{ DTYPE }}_MAX;

        if (input_count >= 2) {
            connx_Tensor* _min = connx_Graph_get(graph, inputs[1]);
            if (_min != NULL) {
                min = *({{TYPE}}*)_min->buffer;
            }

            if (input_count >= 3) {
                connx_Tensor* _max = connx_Graph_get(graph, inputs[2]);
                if (_max != NULL) {
                    max = *({{TYPE}}*)_max->buffer;
                }
            }
        }

        {{TYPE}}* Y_base = ({{TYPE}}*)Y->buffer;
        {{TYPE}}* X_base = ({{TYPE}}*)X->buffer;

        int32_t total = connx_Int32_product(X->ndim, X->shape);
        for (int32_t i = 0; i < total; i++) {
            {{TYPE}} x = *X_base++;
            if (x < min) {
                x = min;
            }

            if (x > max) {
                x = max;
            }

            *Y_base++ = x;
        }

        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("Clip: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
