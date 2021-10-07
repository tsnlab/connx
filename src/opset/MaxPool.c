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

int MaxPool(connx_Graph* graph, uint32_t output_count, uint32_t* outputs, __attribute__((unused)) uint32_t input_count,
            uint32_t* inputs, void** attributes) {
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

    int32_t y_idx = 0;

    int32_t storage_unit = 0;
    if (storage_order == 1) {
        if (feature_dim >= 2) {
            storage_unit = output_shape[feature_dim - 2] * output_shape[feature_dim - 1];
        } else {
            storage_unit = 0;
        }
    }

    // Figure out dilated kernel shape
    int32_t new_kernel_shape[feature_dim];
    for (int32_t i = 0; i < feature_dim; i++) {
        new_kernel_shape[i] = dilations[i] * kernel_shape[i];
        if (dilations[i] != 1) {
            new_kernel_shape[i] -= 1;
        }
    }

    int32_t units[2 + feature_dim]; // batch_unit, channel_unit, feature units

    units[X->ndim - 1] = 1;

    for (int32_t i = X->ndim - 2; i >= 0; i--) {
        units[i] = units[i + 1] * X->shape[i + 1];
    }

    int32_t batch_count = X->shape[0];
    int32_t channel_count = X->shape[1];

    connx_Slice x_patch_slices[X->ndim];

    switch (X->dtype) {
        TEMPLATE_START(UINT8, UINT16, FLOAT32, FLOAT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE INT32
#define TEMPLATE_TYPE int32_t
    case TEMPLATE_DTYPE: {
        TEMPLATE_TYPE* Y_array = (TEMPLATE_TYPE*)Y->buffer;

        for (int32_t batch = 0; batch < batch_count; batch++) {
            connx_Slice_set(&x_patch_slices[0], batch, batch + 1, 1);

            for (int32_t channel = 0; channel < channel_count; channel++) {
                connx_Slice_set(&x_patch_slices[1], channel, channel + 1, 1);

                // x_iter
                connx_Slice x_slices[feature_dim];
                for (int32_t i = 0; i < feature_dim; i++) {
                    connx_Slice_set(&x_slices[i], -pads[i], -pads[i] + output_shape[i] * strides[i], strides[i]);
                }

                connx_Iterator x_iter;
                connx_Iterator_init(&x_iter, feature_dim, x_slices);

                while (connx_Iterator_next(&x_iter, 1)) {
                    TEMPLATE_TYPE y = 0;
                    int64_t argmax_idx = -1;
                    int32_t k_idx[feature_dim];

                    // Make slicers for copy patch of X[batch, channel] on to padded X
                    int32_t x_idxs[feature_dim];
                    connx_Iterator_indices(&x_iter, x_idxs);
                    for (int32_t i = 0; i < feature_dim; i++) {
                        int32_t x_idx = x_idxs[i];
                        int32_t x_start = x_idx < 0 ? 0 : x_idx;
                        int32_t end = x_idx + new_kernel_shape[i];
                        int32_t x_end = feature_shape[i] < end ? feature_shape[i] : end;
                        connx_Slice_set(&x_patch_slices[i + 2], x_start, x_end, dilations[i]);

                        int32_t x_padded_start = x_idx < 0 ? -x_idx : 0;
                        k_idx[i] = x_padded_start;
                    }

                    // Get patch of X[batch, channel]
                    connx_Tensor* x_patch = connx_Tensor_get_by_slice(X, x_patch_slices);
                    int32_t tmp_size = connx_Int32_product(feature_dim, x_patch->shape + 2);
                    int32_t kernel_offset = connx_TEMPLATE_NAME_argmax(tmp_size, &y, x_patch->buffer);

                    // Convert offset to index
                    for (int32_t i = 1; i < feature_dim; i++) {
                        int32_t unit = connx_Int32_product(feature_dim - i, x_patch->shape + 2 + i);
                        k_idx[i - 1] += kernel_offset / unit;
                        kernel_offset = kernel_offset % unit;
                    }
                    k_idx[feature_dim - 1] += kernel_offset;

                    int32_t d_idx[feature_dim];
                    int32_t d_offset = 0;

                    // Compute offset of max argmax in X[batch, channel]
                    for (int32_t i = 0; i < feature_dim; i++) {
                        d_idx[i] = x_idxs[i] + k_idx[i];
                        d_offset += d_idx[i] * units[2 + i];
                    }

                    argmax_idx = d_offset;
                    Y_array[y_idx] = y;

                    if (output_count > 1) {
                        if (storage_order == 1) {
                            int32_t remainder = y_idx % storage_unit;
                            int32_t share = y_idx - remainder;
                            int32_t a = remainder * output_shape[feature_dim - 1];
                            Indices_array[share + a % storage_unit + (int32_t)(a / storage_unit)] = argmax_idx;
                        } else {
                            Indices_array[y_idx] = argmax_idx;
                        }
                    }
                    y_idx++;
                    connx_Tensor_unref(x_patch);
                }
            }
        }
        break;
    }
        TEMPLATE_END()
    default:
        connx_error("MaxPool: Datatype %d is not supported yet.\n", X->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);
    if (output_count > 1) {
        connx_Graph_set(graph, outputs[1], Indices);
    }

    return CONNX_OK;
}
