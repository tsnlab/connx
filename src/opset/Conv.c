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
#include <string.h>
#include <strings.h> // bzero

#include <connx/accel.h>
#include <connx/connx.h>

TEMPLATE_START(FLOAT32, FLOAT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
#define connx_TEMPLATE_NAME_add connx_Float32_add
static void _conv_TEMPLATE_NAME(connx_Tensor* Y, int32_t y_idx, connx_Tensor* X, connx_Iterator* x_iter,
                                connx_Tensor* W, int32_t batch, int32_t x_channel, int32_t w_channel,
                                int32_t feature_map, int32_t* dilations) {

    int32_t feature_dim = X->ndim - 2;
    int32_t* feature_shape = X->shape + 2;

    int32_t kernel_dim = W->ndim - 2;
    int32_t* kernel_shape = W->shape + 2;
    int32_t kernel_size = connx_Int32_product(kernel_dim, kernel_shape);

    // Calculate weight offset
    connx_Slice w_slices[W->ndim];
    connx_Slice_set(&w_slices[0], feature_map, feature_map + 1, 1);
    connx_Slice_set(&w_slices[1], w_channel, w_channel + 1, 1);

    for (int32_t i = 0; i < kernel_dim; i++) {
        connx_Slice_set(&w_slices[2 + i], 0, kernel_shape[i], 1);
    }

    connx_Iterator w_iter;
    connx_Iterator_init(&w_iter, W->ndim, w_slices);
    connx_Iterator_next(&w_iter, 1);

    int32_t data_size = connx_DataType_size(TEMPLATE_DTYPE);
    int32_t w_offset = connx_Iterator_offset(&w_iter, W->shape) * data_size;

    // Make slice of X(x = X[batch, x_channel])
    connx_Slice x_slices[X->ndim];
    connx_Slice_set(&x_slices[0], batch, batch + 1, 1);
    connx_Slice_set(&x_slices[1], x_channel, x_channel + 1, 1);

    connx_Tensor* x_patch = connx_Tensor_alloc(X->dtype, kernel_dim, kernel_shape);

    TEMPLATE_TYPE* Y_flatten = (TEMPLATE_TYPE*)Y->buffer;

    while (connx_Iterator_next(x_iter, 1)) {
        TEMPLATE_TYPE y = 0;

        // Clear x patch
        bzero(x_patch->buffer, x_patch->size);

        // Make a slice for copying patch of X[batch, channel] to x_patch
        connx_Slice x_patch_slices[feature_dim];
        int32_t x_idxs[feature_dim];
        connx_Iterator_indices(x_iter, x_idxs);
        for (int32_t i = 0; i < feature_dim; i++) {
            int32_t x_idx = x_idxs[i];
            int32_t x_start = x_idx < 0 ? -x_idx % dilations[i] : x_idx;
            int32_t end = x_idx + (kernel_shape[i] - 1) * dilations[i] + 1;
            int32_t x_end = feature_shape[i] < end ? feature_shape[i] : end;
            connx_Slice_set(&x_slices[i + 2], x_start, x_end, dilations[i]);

            int32_t x_patch_start = x_idx < 0 ? -x_idx : 0;
            int32_t x_patch_end = x_patch_start + (x_end - x_start + dilations[i] - 1) / dilations[i];
            connx_Slice_set(&x_patch_slices[i], x_patch_start, x_patch_end, 1);
        }

        // Get slice of X. X[tuple(x_slices)]. x_patch[x_patch_slices] = X[tuple(x_slices)]
        connx_Tensor_set_by_slice(x_patch, x_patch_slices, X, x_slices);

        // Convolute
        y = connx_TEMPLATE_NAME_mul_and_sum(kernel_size, x_patch->buffer, W->buffer + w_offset);

        Y_flatten[y_idx] += y;
        y_idx++;
    }

    connx_Tensor_unref(x_patch);
}
TEMPLATE_END()

int Conv(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs, uint32_t input_count,
         uint32_t* inputs, void** attributes) {
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
    int32_t kernel_dim = _kernel_shape->count;

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

    // init w_iter
    connx_Slice w_slices[kernel_dim];
    for (int32_t i = 0; i < kernel_dim; i++) {
        connx_Slice_set(&w_slices[i], 0, kernel_shape[i], 1);
    }

    connx_Iterator w_iter;
    connx_Iterator_init(&w_iter, kernel_dim, w_slices);

    switch (X->dtype) {
        TEMPLATE_START(FLOAT32, FLOAT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
#define connx_TEMPLATE_NAME_add connx_Float32_add
#define connx_TEMPLATE_NAME_broadcast connx_Float32_broadcast
    case TEMPLATE_DTYPE: {
        TEMPLATE_TYPE* Y_flatten = (TEMPLATE_TYPE*)Y->buffer;
        TEMPLATE_TYPE* B_flatten = NULL;
        if (B != NULL) {
            B_flatten = (TEMPLATE_TYPE*)B->buffer;
        }

        int32_t batch_count = X->shape[0];
        int32_t channel_count = W->shape[1];
        int32_t feature_group = W->shape[0] / group;

        int32_t y_idx = 0;
        int32_t y_unit = connx_Int32_product(feature_dim, output_shape);

        for (int32_t batch = 0; batch < batch_count; batch++) {
            for (int32_t g = 0; g < group; g++) {
                for (int32_t feature_map = g * feature_group; feature_map < (g + 1) * feature_group; feature_map++) {
                    for (int32_t channel = 0; channel < channel_count; channel++) {
                        _conv_TEMPLATE_NAME(Y, y_idx, X, &x_iter, W, batch, g * channel_count + channel, channel,
                                            feature_map, dilations);
                    }

                    if (B_flatten != NULL) {
                        TEMPLATE_TYPE B_array[y_unit];
                        connx_TEMPLATE_NAME_broadcast(y_unit, B_array, 1, B_flatten + feature_map);
                        connx_TEMPLATE_NAME_add(y_unit, Y_flatten + y_idx, Y_flatten + y_idx, B_array);
                    }

                    y_idx += y_unit;
                }
            }
        }

        break;
    }
        TEMPLATE_END()
    default:
        connx_error("Conv: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
