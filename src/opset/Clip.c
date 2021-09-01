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
#include <connx/accel.h>
#include <connx/connx.h>

TEMPLATE_START(FLOAT16, FLOAT32, FLOAT64, UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
#define TEMPLATE_DTYPE_MIN FLOAT32
TEMPLATE_TYPE _clip_TEMPLATE_NAME(TEMPLATE_TYPE x, TEMPLATE_TYPE min_val, TEMPLATE_TYPE max_val) {
    if (x < min_val) {
        x = min_val;
    }

    if (x > max_val) {
        x = max_val;
    }

    return x;
}
TEMPLATE_END()

int Clip(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
         __attribute__((unused)) uint32_t input_count, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    // input
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* min = NULL;
    connx_Tensor* max = NULL;

    if (input_count >= 2) {
        min = connx_Graph_get(graph, inputs[1]);
        if (input_count == 3) {
            max = connx_Graph_get(graph, inputs[2]);
        }
    }

    connx_Tensor* Y = connx_Tensor_alloc(X->dtype, X->ndim, X->shape);

    switch (X->dtype) {
        TEMPLATE_START(FLOAT16, FLOAT32, FLOAT64, UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
#define TEMPLATE_DTYPE_MIN FLOAT32
#define TEMPLATE_DTYPE_MAX FLOAT32
    case TEMPLATE_DTYPE: {
        TEMPLATE_TYPE* Y_base = (TEMPLATE_TYPE*)Y->buffer;
        TEMPLATE_TYPE* X_base = (TEMPLATE_TYPE*)X->buffer;

        TEMPLATE_TYPE min_val = TEMPLATE_DTYPE_MIN;
        if (min != NULL) {
            min_val = *(TEMPLATE_TYPE*)min->buffer;
        }

        TEMPLATE_TYPE max_val = TEMPLATE_DTYPE_MAX;
        if (max != NULL) {
            max_val = *(TEMPLATE_TYPE*)max->buffer;
        }

        int32_t total = 1;
        for (int32_t i = 0; i < X->ndim; i++) {
            total *= X->shape[i];
        }

        for (int32_t i = 0; i < total; i++) {
            TEMPLATE_TYPE x = *X_base++;
            x = _clip_TEMPLATE_NAME(x, min_val, max_val);
            *Y_base++ = x;
        }

        break;
    }
        TEMPLATE_END()
    default:
        connx_error("Clip: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
