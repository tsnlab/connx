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
#include <math.h>
#include <string.h>  // memset
#include <strings.h> // bzero

#include <connx/accel.h>
#include <connx/connx.h>

int GlobalAveragePool(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                      __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                      __attribute__((unused)) void** attributes) {
    // input
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);

    int32_t batch_count = X->shape[0];
    int32_t channel_count = X->shape[1];
    int32_t unit = 1;
    for (int32_t i = 2; i < X->ndim; i++) {
        unit *= X->shape[i];
    }

    // output
    int32_t Y_shape[X->ndim];
    for (int i = 0; i < X->ndim; i++) {
        Y_shape[i] = 1;
    }
    Y_shape[0] = X->shape[0];
    Y_shape[1] = X->shape[1];
    connx_Tensor* Y = connx_Tensor_alloc(X->dtype, X->ndim, Y_shape);

    switch (X->dtype) {
        TEMPLATE_START(FLOAT32, FLOAT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
    case TEMPLATE_DTYPE: {
        TEMPLATE_TYPE* Y_base = (TEMPLATE_TYPE*)Y->buffer;
        TEMPLATE_TYPE* X_base = (TEMPLATE_TYPE*)X->buffer;

        for (int32_t batch = 0; batch < batch_count; batch++) {
            for (int32_t channel = 0; channel < channel_count; channel++) {
                TEMPLATE_TYPE average = 0;
                for (int32_t i = 0; i < unit; i++) {
                    average += (*X_base++) / unit;
                }

                *Y_base++ = average;
            }
        }
        break;
    }
        TEMPLATE_END()
    default:
        connx_error("GlobalAveragePool: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
