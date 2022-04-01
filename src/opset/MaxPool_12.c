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
#include <strings.h> // bzero

#include <connx/accel.h>
#include <connx/connx.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*{% for DTYPE, TYPE in loop_types(UINT8, UINT16, FLOAT32, FLOAT64) %}*/
// clang-format off
static int32_t max_pool_1d_{{DTYPE}}({{TYPE}}* Y_flatten, int32_t* Y_shape, int64_t* Indices_flatten,
                                     {{TYPE}}* X_flatten, int32_t* X_shape, int32_t* pads, int32_t* kernel_shape,
                                     int32_t* dilations, int32_t* strides) {
    // clang-format on

    int32_t y_idx = 0;
    int32_t Y_shape0 = Y_shape[0];
    int32_t X_shape0 = X_shape[0];
    int32_t kernel_shape0 = kernel_shape[0];
    int32_t strides0 = strides[0];
    int32_t dilations0 = dilations[0];

    // X iteration
    for (int32_t x_idx0 = -pads[0]; x_idx0 < -pads[0] + Y_shape0 * strides0; x_idx0 += strides0) {
        // kernel iteration, p means patch
        // clang-format off
        {{TYPE}} y = 0;
        // clang-format on
        int32_t argmax_offset = -1;

        for (int32_t p_idx0 = MAX(x_idx0, 0); p_idx0 < MIN(x_idx0 + kernel_shape0 * dilations0, X_shape0);
             p_idx0 += dilations0) {

            int32_t x_offset = p_idx0;
            // clang-format off
            {{TYPE}} p = X_flatten[x_offset];
            // clang-format on

            if (argmax_offset < 0 || p > y) { // p > y is max pool
                y = p;
                argmax_offset = x_offset;
            }
        }

        if (Indices_flatten != NULL) {
            Indices_flatten[y_idx] = argmax_offset;
        }

        Y_flatten[y_idx++] = y;
    }

    return y_idx;
}

// clang-format off
static int32_t max_pool_2d_{{DTYPE}}({{TYPE}}* Y_flatten, int32_t* Y_shape, int64_t* Indices_flatten,
                                     {{TYPE}}* X_flatten, int32_t* X_shape, int32_t* pads, int32_t* kernel_shape,
                                     int32_t* dilations, int32_t* strides) {
    // clang-format on

    int32_t y_idx = 0;
    int32_t Y_shape0 = Y_shape[0];
    int32_t Y_shape1 = Y_shape[1];
    int32_t X_shape0 = X_shape[0];
    int32_t X_shape1 = X_shape[1];
    int32_t kernel_shape0 = kernel_shape[0];
    int32_t kernel_shape1 = kernel_shape[1];
    int32_t strides0 = strides[0];
    int32_t strides1 = strides[1];
    int32_t dilations0 = dilations[0];
    int32_t dilations1 = dilations[1];

    // X iteration
    for (int32_t x_idx0 = -pads[0]; x_idx0 < -pads[0] + Y_shape0 * strides0; x_idx0 += strides0) {
        for (int32_t x_idx1 = -pads[1]; x_idx1 < -pads[1] + Y_shape1 * strides1; x_idx1 += strides1) {
            // kernel iteration, p means patch
            // clang-format off
            {{TYPE}} y = 0;
            // clang-format on
            int32_t argmax_offset = -1;

            for (int32_t p_idx0 = MAX(x_idx0, 0); p_idx0 < MIN(x_idx0 + kernel_shape0 * dilations0, X_shape0);
                 p_idx0 += dilations0) {

                for (int32_t p_idx1 = MAX(x_idx1, 0); p_idx1 < MIN(x_idx1 + kernel_shape1 * dilations1, X_shape1);
                     p_idx1 += dilations1) {

                    int32_t x_offset = p_idx0 * X_shape1 + p_idx1;
                    // clang-format off
                    {{TYPE}} p = X_flatten[x_offset];
                    // clang-format on

                    if (argmax_offset < 0 || p > y) { // p > y is max pool
                        y = p;
                        argmax_offset = x_offset;
                    }
                }
            }

            if (Indices_flatten != NULL) {
                Indices_flatten[y_idx] = argmax_offset;
            }

            Y_flatten[y_idx++] = y;
        }
    }

    return y_idx;
}
/*{% endfor %}*/

// clang-format off
int MaxPool_{{op_version}}(connx_Graph* graph, uint32_t output_count, uint32_t* outputs,
                            // clang-format on
                            __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
                            __attribute__((unused)) uint32_t attribute_count, void** attributes) {
    // inputs
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);

    // attributes
    char* auto_pad = attributes[0];
    int32_t ceil_mode = *(int32_t*)attributes[1];
    connx_AttributeInts* _dilations = attributes[2];
    connx_AttributeInts* _kernel_shape = attributes[3];
    connx_AttributeInts* _pads = attributes[4];
    int32_t storage_order = *(int32_t*)attributes[5];
    connx_AttributeInts* _strides = attributes[6];

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
    bzero(output_shape, sizeof(int32_t) * feature_dim);

    if (strcmp(auto_pad, "NOTSET") == 0) {
        if (ceil_mode == 0) {
            for (int i = 0; i < feature_dim; i++) {
                output_shape[i] =
                    (feature_shape[i] + pads[i] + pads[i + feature_dim] - ((kernel_shape[i] - 1) * dilations[i] + 1)) /
                        strides[i] +
                    1;
            }
        } else {
            for (int i = 0; i < feature_dim; i++) {
                output_shape[i] = ceilf((float)(feature_shape[i] + pads[i] + pads[i + feature_dim] -
                                                ((kernel_shape[i] - 1) * dilations[i] + 1)) /
                                            strides[i] +
                                        1);
            }
        }
    } else if (strcmp(auto_pad, "VALID") == 0) {
        for (int i = 0; i < feature_dim; i++) {
            output_shape[i] =
                ceilf((float)(feature_shape[i] - ((kernel_shape[i] - 1) * dilations[i] + 1) + 1) / strides[i]);
        }
    } else if (strcmp(auto_pad, "SAME_UPPER") == 0) {
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
    }

    // MaxPool
    int32_t Y_shape[2 + feature_dim];
    Y_shape[0] = X->shape[0];
    Y_shape[1] = X->shape[1];
    memcpy(Y_shape + 2, output_shape, sizeof(int32_t) * feature_dim);

    connx_Tensor* Y = connx_Tensor_alloc(X->dtype, 2 + feature_dim, Y_shape);
    connx_Tensor* Indices = NULL;
    int64_t* Indices_array = NULL;
    if (output_count > 1) {
        Indices = connx_Tensor_alloc(CONNX_INT64, 2 + feature_dim, Y_shape);
        Indices_array = (int64_t*)Indices->buffer;
    }

    int32_t batch_count = X->shape[0];
    int32_t channel_count = X->shape[1];
    int32_t X_unit = connx_Int32_product(feature_dim, X->shape + 2);

    switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(UINT8, UINT16, FLOAT32, FLOAT64) %}*/
    case {{ DTYPE }}: {
        {{TYPE}}* Y_flatten = Y->buffer;
        {{TYPE}}* X_flatten = X->buffer;
        int32_t X_offset = 0;

        for (int32_t batch = 0; batch < batch_count; batch++) {
            for (int32_t channel = 0; channel < channel_count; channel++) {
                switch (feature_dim) {
                case 1:
                    // clang-format off
                    Y_flatten += max_pool_1d_{{DTYPE}}(Y_flatten, output_shape, Indices_array, X_flatten,
                                                       feature_shape, pads, kernel_shape, dilations, strides);
                    // clang-format on
                    break;
                case 2:
                    // clang-format off
                    Y_flatten += max_pool_2d_{{DTYPE}}(Y_flatten, output_shape, Indices_array, X_flatten,
                                                       feature_shape, pads, kernel_shape, dilations, strides);
                    // clang-format on
                    break;
                default:
                    connx_error("MaxPool: Feature dimension %d is not supported yet.\n", feature_dim);
                    return CONNX_NOT_SUPPORTED_DATATYPE;
                }

                X_offset += X_unit;
                X_flatten += X_unit;
            }
        }
        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("MaxPool: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    // Transpose Indices if storage_order == 1
    if (storage_order == 1 && feature_dim >= 2) {
        int32_t height = output_shape[feature_dim - 2];
        int32_t width = output_shape[feature_dim - 1];
        int32_t unit = height * width;

        int64_t matrix[height][width];
        int64_t* array = Indices_array;
        for (int32_t batch = 0; batch < batch_count; batch++) {
            for (int32_t channel = 0; channel < channel_count; channel++) {
                memcpy(matrix, array, height * width * sizeof(int64_t));

                int32_t i = 0;
                for (int32_t w = 0; w < width; w++) {
                    for (int32_t h = 0; h < height; h++) {
                        array[i++] = matrix[h][w];
                    }
                }

                array += unit;
            }
        }
    }

    connx_Graph_set(graph, outputs[0], Y);
    if (output_count > 1) {
        connx_Graph_set(graph, outputs[1], Indices);
    }

    return CONNX_OK;
}
