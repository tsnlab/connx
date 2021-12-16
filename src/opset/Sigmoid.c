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

#include <connx/accel.h>
#include <connx/connx.h>

int Sigmoid(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
            __attribute__((unused)) uint32_t input_count, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* Y = connx_Tensor_alloc_like(X);
    if (Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t total = connx_Int32_product(X->ndim, X->shape);

    switch (X->dtype) {
        TEMPLATE_START(FLOAT32, FLOAT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
    case TEMPLATE_DTYPE: {
        TEMPLATE_TYPE* X_array = X->buffer;
        TEMPLATE_TYPE* Y_array = Y->buffer;

        for (int32_t i = 0; i < total; i++) {
#if TEMPLATE_DTYPE == FLOAT32
            Y_array[i] = 1 / (1 + expf(-X_array[i]));
#else
            Y_array[i] = 1 / (1 + exp(-X_array[i]));
#endif // TEMPLATE_DTYPE == FLOAT32
        }
        break;
    }
        TEMPLATE_END()
    default:
        connx_error("Sigmoid: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
