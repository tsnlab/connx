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
#include <string.h>  // memset
#include <strings.h> // bzero

#include <connx/accel.h>
#include <connx/connx.h>

// clang-format off
int BatchNormalization_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count,
                                       // clang-format on
                                       uint32_t* outputs, __attribute__((unused)) uint32_t input_count,
                                       uint32_t* inputs, __attribute__((unused)) uint32_t attribute_count,
                                       void** attributes) {
    // input
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* scale = connx_Graph_get(graph, inputs[1]);
    connx_Tensor* B = connx_Graph_get(graph, inputs[2]);
    connx_Tensor* mean = connx_Graph_get(graph, inputs[3]);
    connx_Tensor* var = connx_Graph_get(graph, inputs[4]);

    // attribute
    float epsilon = *(float*)attributes[0];

    int32_t batch_count = X->shape[0];
    int32_t channel_count = X->shape[1];
    int32_t unit = 1;
    for (int32_t i = 2; i < X->ndim; i++) {
        unit *= X->shape[i];
    }

    connx_Tensor* Y = connx_Tensor_alloc(X->dtype, X->ndim, X->shape);

    switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(FLOAT32, FLOAT64) %}*/
    case {{ DTYPE }}: {
        {{TYPE}}* Y_base = ({{TYPE}}*)Y->buffer;
        {{TYPE}}* X_base = ({{TYPE}}*)X->buffer;

        {{TYPE}}* scales = ({{TYPE}}*)scale->buffer;
        {{TYPE}}* Bs = ({{TYPE}}*)B->buffer;
        {{TYPE}}* means = ({{TYPE}}*)mean->buffer;
        {{TYPE}}* vars = ({{TYPE}}*)var->buffer;

        for (int32_t batch = 0; batch < batch_count; batch++) {
            for (int32_t channel = 0; channel < channel_count; channel++) {
                // clang-format off
                {{TYPE}} scale_value = scales[channel];
                {{TYPE}} B_value = Bs[channel];
                {{TYPE}} mean_value = means[channel];
                {{TYPE}} sqrt_value = sqrtf(vars[channel] + epsilon);
                {{TYPE}} scale_div_sqrt_value = scale_value / sqrt_value;
                // clang-format on

                for (int32_t i = 0; i < unit; i++) {
                    *Y_base++ = (*X_base++ - mean_value) * scale_div_sqrt_value + B_value;
                }
            }
        }
        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("BatchNormalization: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
