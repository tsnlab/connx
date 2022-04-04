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
#include <string.h>

#include <connx/accel.h>
#include <connx/connx.h>

/*{% for DTYPE, TYPE in loop_types(FLOAT32, FLOAT64) %}*/
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#include <stdio.h>
// clang-format off
static void conv_1d_{{DTYPE}}({{TYPE}}* Y_flatten, int32_t* Y_shape, {{TYPE}}* X_flatten, int32_t* X_shape, {{TYPE}}* W_flatten, int32_t* W_shape, int32_t* pads, int32_t* dilations, int32_t* strides) {
    // clang-format on

    int32_t Y_shape0 = Y_shape[0];
    int32_t X_shape0 = X_shape[0];
    int32_t W_shape0 = W_shape[0];
    int32_t strides0 = strides[0];
    int32_t dilations0 = dilations[0];

    // X iteration
    int32_t y_idx = 0;
    for (int32_t x_idx0 = -pads[0]; x_idx0 < -pads[0] + Y_shape0 * strides0; x_idx0 += strides0) {
        
        {{TYPE}} y = 0;

        // kernel iteration, p means patch
        for (int32_t p_idx0 = MAX(x_idx0, 0), w_idx0 = p_idx0 - x_idx0;
             p_idx0 < MIN(x_idx0 + W_shape0 * dilations0, X_shape0);
             p_idx0 += dilations0, w_idx0++) {

            int32_t x_offset = p_idx0;
            int32_t w_offset = w_idx0;

            // clang-format off
            {{TYPE}} p = X_flatten[x_offset];
            {{TYPE}} w = W_flatten[w_offset];
            // clang-format on

            y += p * w;
        }

        Y_flatten[y_idx++] += y;
    }
}

/*{% endfor %}*/

// clang-format off
int Conv_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                         // clang-format on
                         uint32_t input_count, uint32_t* inputs, __attribute__((unused)) uint32_t attribute_count,
                         void** attributes) {
    // inputs
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* W = connx_Graph_get(graph, inputs[1]);
    connx_Tensor* B = NULL;
    if (input_count >= 3) {
        B = connx_Graph_get(graph, inputs[2]);
    }

    // attributes
    char* auto_pad = attributes[0];
    connx_AttributeInts* _dilations = attributes[1];
    int32_t group = *(int32_t*)attributes[2];
    connx_AttributeInts* _kernel_shape = attributes[3];
    connx_AttributeInts* _pads = attributes[4];
    connx_AttributeInts* _strides = attributes[5];

    // feature dimension
    int32_t feature_dim = X->ndim - 2;
    int32_t* feature_shape = X->shape + 2;

    // default attribute setting
    int32_t dilations[feature_dim];
    if (_dilations->count == 0) {
        for (int32_t i = 0; i < feature_dim; i++) {
            dilations[i] = 1;
        }
    } else {
        memcpy(dilations, _dilations->array, sizeof(int32_t) * feature_dim);
    }

    int32_t* kernel_shape = _kernel_shape->array;

    int32_t pads[feature_dim * 2];
    if (_pads->count == 0) {
        for (int32_t i = 0; i < feature_dim * 2; i++) {
            pads[i] = 0;
        }
    } else {
        memcpy(pads, _pads->array, sizeof(int32_t) * feature_dim * 2);
    }

    int32_t strides[feature_dim];
    if (_strides->count == 0) {
        for (int32_t i = 0; i < feature_dim; i++) {
            strides[i] = 1;
        }
    } else {
        memcpy(strides, _strides->array, sizeof(int32_t) * feature_dim);
    }

    // output_spatial_shape
    int32_t output_shape[feature_dim];

    if (strcmp(auto_pad, "SAME_UPPER") == 0) {
        for (int i = 0; i < feature_dim; i++) {
            output_shape[i] = ceilf((float)feature_shape[i] / strides[i]);
            int32_t pad =
                (output_shape[i] - 1) * strides[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - feature_shape[i];
            pads[i] = pads[i + feature_dim] = pad / 2;
            if (pad % 2 == 1) {
                pads[i + feature_dim]++;
            }
        }
    } else if (strcmp(auto_pad, "SAME_LOWER") == 0) {
        for (int i = 0; i < feature_dim; i++) {
            output_shape[i] = ceilf((float)feature_shape[i] / strides[i]);
            int32_t pad =
                (output_shape[i] - 1) * strides[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - feature_shape[i];
            pads[i] = pads[i + feature_dim] = pad / 2;
            if (pad % 2 == 1) {
                pads[i]++;
            }
        }
    } else {
        for (int i = 0; i < feature_dim; i++) {
            output_shape[i] =
                (feature_shape[i] + pads[i] + pads[i + feature_dim] - ((kernel_shape[i] - 1) * dilations[i] + 1)) /
                    strides[i] +
                1;
        }
    }

    // Conv
    int32_t Y_shape[2 + feature_dim];
    Y_shape[0] = X->shape[0];
    Y_shape[1] = W->shape[0];
    memcpy(Y_shape + 2, output_shape, sizeof(int32_t) * feature_dim);

    connx_Tensor* Y = connx_Tensor_alloc(X->dtype, 2 + feature_dim, Y_shape);

    // init x_iter
    connx_Slice x_slices[feature_dim];
    for (int32_t i = 0; i < feature_dim; i++) {
        connx_Slice_set(&x_slices[i], -pads[i], -pads[i] + output_shape[i] * strides[i], strides[i]);
    }

    connx_Iterator x_iter;
    connx_Iterator_init(&x_iter, feature_dim, x_slices);

    switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(FLOAT32, FLOAT64) %}*/
    case {{DTYPE}}: {
        int32_t batch_count = X->shape[0];
        int32_t feature_count = W->shape[0];
        int32_t channel_count = W->shape[1];
        int32_t feature_group = feature_count / group;

        // clang-format off
        {{TYPE}}* X_flatten = ({{TYPE}}*)X->buffer;
        {{TYPE}}* Y_flatten = ({{TYPE}}*)Y->buffer;
        {{TYPE}}* W_flatten = ({{TYPE}}*)W->buffer;
        {{TYPE}}* B_flatten = ({{TYPE}}*)B->buffer;
        // clang-format on

        int32_t Y_unit = connx_Int32_product(Y->ndim - 2, Y->shape + 2);
        int32_t X_unit = connx_Int32_product(X->ndim - 2, X->shape + 2);
        int32_t W_unit = connx_Int32_product(W->ndim - 2, W->shape + 2);

        switch (feature_dim) {
        case 1:
            for (int32_t batch = 0; batch < batch_count; batch++) {
                for (int32_t feature = 0; feature < feature_count; feature++) {
                    int32_t g = feature / feature_group;

                    for (int32_t channel = 0; channel < channel_count; channel++) {

                        //fprintf(stderr, "batch: %d, feature: %d, group: %d, channel: %d\n", batch, feature, g, channel);
                        {{TYPE}}* X_array = X_flatten + X_unit * (channel_count * g + channel);
                        {{TYPE}}* W_array = W_flatten + W_unit * (channel_count * feature + channel);

                        /*
                        fprintf(stderr, "\tX: ");
                        for (int32_t i = 0; i < 3; i++) {
                            fprintf(stderr, "%f ", X_array[i]);
                        }
                        fprintf(stderr, "\n");
                        */
                        /*
                        fprintf(stderr, "\tW: ");
                        for (int32_t i = 0; i < 3; i++) {
                            fprintf(stderr, "%f ", W_array[i]);
                        }
                        fprintf(stderr, "\n");
                        */

                        conv_1d_{{DTYPE}}(Y_flatten, Y_shape + 2, X_array, feature_shape, W_array, kernel_shape, pads, dilations, strides);
                    }

                    if (B_flatten != NULL) {
                        {{TYPE}} bias = B_flatten[feature];
                        //fprintf(stderr, "bias: %f\n", (float)bias);

                        for (int32_t i = 0; i < Y_unit; i++) {
                            Y_flatten[i] += bias;
                        }
                    }
                    /*
                    for (int32_t i = 0; i < Y_unit; i++) {
                        fprintf(stderr, "%f ", Y_flatten[i]);
                    }
                    fprintf(stderr, "\n");
                    */

                    Y_flatten += Y_unit;
                }
                X_flatten += X_unit * channel_count * group;
            }

            break;
        default:
            connx_error("Conv: dimension is not supported yet: %d\n", feature_dim);
            return CONNX_NOT_SUPPORTED_DATATYPE;
        }

        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("Conv: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
