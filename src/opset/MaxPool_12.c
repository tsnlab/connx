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

#include <stdio.h>

/*{% for DTYPE, TYPE in loop_types(UINT8, UINT16, FLOAT32, FLOAT64) %}*/
static int32_t max_pool_1d_{{DTYPE}}({{TYPE}}* Y_flatten, int32_t* Y_shape, {{TYPE}}* X_flatten, int32_t* X_shape,
                                     int32_t* pads, int32_t* kernel_shape, int32_t* dilations, int32_t* strides) {

    int32_t y_idx = 0;
    int32_t Y_shape0 = Y_shape[0];
    int32_t X_shape0 = X_shape[0];
    int32_t kernel_shape0 = kernel_shape[0];
    int32_t strides0 = strides[0];
    int32_t dilations0 = dilations[0];

    // X iteration
    //fprintf(stderr, "x: %d ~ %d\n", -pads[0], X_shape0 + pads[1] - (kernel_shape0 - 1));
    //for (int32_t x_idx0 = -pads[0]; x_idx0 < X_shape0 + pads[1] - (kernel_shape0 - 1); x_idx0 += strides0) {
    for (int32_t x_idx0 = -pads[0]; x_idx0 < -pads[0] + Y_shape0 * strides0; x_idx0 += strides0) {
        // kernel iteration, p means patch
        {{TYPE}} y = 0;
        int32_t argmax_offset = -1;

        //fprintf(stderr, "p: %d ~ %d (%d)\n", MAX(x_idx0, 0), MIN(x_idx0 + kernel_shape0, X_shape0), MIN(x_idx0 + kernel_shape0, X_shape0) - MAX(x_idx0, 0));
        for (int32_t p_idx0 = MAX(x_idx0, 0); p_idx0 < MIN(x_idx0 + kernel_shape0 * dilations0, X_shape0); p_idx0 += dilations0) {
            int32_t x_offset = p_idx0;
            {{TYPE}} p = X_flatten[x_offset];

            if (argmax_offset < 0 || p > y) { // p > y is max pool
                y = p;
                argmax_offset = x_offset;
            }
        }

        Y_flatten[y_idx++] = y;
    }
    //fprintf(stderr, "y_idx: %d\n", y_idx);

    return y_idx;
}

static int32_t max_pool_2d_{{DTYPE}}({{TYPE}}* Y_flatten, int32_t* Y_shape, {{TYPE}}* X_flatten, int32_t* X_shape,
                                     int32_t* pads, int32_t* kernel_shape, int32_t* dilations, int32_t* strides) {

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
    //fprintf(stderr, "x_idx0: %d ~ %d\n", -pads[0], X_shape0 + pads[2] - (kernel_shape0 - 1));
    for (int32_t x_idx0 = -pads[0]; x_idx0 < -pads[0] + Y_shape0 * strides0; x_idx0 += strides0) {
        //fprintf(stderr, "x_idx1: %d ~ %d\n", -pads[1], X_shape1 + pads[3] - (kernel_shape1 - 1));
        for (int32_t x_idx1 = -pads[1]; x_idx1 < -pads[1] + Y_shape1 * strides1; x_idx1 += strides1) {
            // kernel iteration, p means patch
            {{TYPE}} y = 0;
            int32_t argmax_offset = -1;

            //fprintf(stderr, "p_idx0: %d ~ %d (%d)\n", MAX(x_idx0, 0), MIN(x_idx0 + kernel_shape0 * dilations0, X_shape0), MIN(x_idx0 + kernel_shape0, X_shape0) - MAX(x_idx0, 0));
            for (int32_t p_idx0 = MAX(x_idx0, 0); p_idx0 < MIN(x_idx0 + kernel_shape0 * dilations0, X_shape0); p_idx0 += dilations0) {
                //fprintf(stderr, "p_idx1: %d ~ %d (%d)\n", MAX(x_idx1, 0), MIN(x_idx1 + kernel_shape1 * dilations1, X_shape1), MIN(x_idx1 + kernel_shape1, X_shape1) - MAX(x_idx1, 0));
                for (int32_t p_idx1 = MAX(x_idx1, 0); p_idx1 < MIN(x_idx1 + kernel_shape1 * dilations1, X_shape1); p_idx1 += dilations1) {
                    int32_t x_offset = p_idx0 * X_shape1 + p_idx1;
                    {{TYPE}} p = X_flatten[x_offset];

                    if (argmax_offset < 0 || p > y) { // p > y is max pool
                        y = p;
                        argmax_offset = x_offset;
                    }
                }
            }

            //fprintf(stderr, "y_idx: %d ", y_idx);
            //fflush(stderr);
            Y_flatten[y_idx++] = y;
        }
    }
    //fprintf(stderr, "y_idx: %d\n", y_idx);

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
                    Y_flatten += max_pool_1d_{{DTYPE}}(Y_flatten, output_shape, X_flatten, feature_shape, pads, kernel_shape, dilations, strides);

                    X_offset += X_unit;
                    X_flatten += X_unit;
                    continue;
                case 2:
                    Y_flatten += max_pool_2d_{{DTYPE}}(Y_flatten, output_shape, X_flatten, feature_shape, pads, kernel_shape, dilations, strides);

                    X_offset += X_unit;
                    X_flatten += X_unit;
                    continue;
                default:
                    connx_error("MaxPool: Feature dimension %d is not supported yet.\n", feature_dim);
                    return CONNX_NOT_SUPPORTED_DATATYPE;
                }

                // x_iter
                connx_Slice x_slices[feature_dim];
                for (int32_t i = 0; i < feature_dim; i++) {
                    connx_Slice_set(&x_slices[i], -pads[i], -pads[i] + output_shape[i] * strides[i], strides[i]);
                }

                connx_Iterator x_iter;
                connx_Iterator_init(&x_iter, feature_dim, x_slices);

                while (connx_Iterator_next(&x_iter, 1)) {
                    // clang-format off
                    {{TYPE}} y = 0;
                    // clang-format on
                    int64_t argmax_idx = -1;
                    int32_t k_idx[feature_dim];

                    // x_patch_iter
                    connx_Slice x_patch_slices[feature_dim];

                    int32_t x_idxs[feature_dim];
                    connx_Iterator_indices(&x_iter, x_idxs);

                    for (int32_t i = 0; i < feature_dim; i++) {
                        int32_t x_patch_start = x_idxs[i];
                        int32_t x_patch_end = x_idxs[i] + kernel_shape[i] * dilations[i];
                        int32_t x_patch_step = dilations[i];

                        if (x_patch_start < 0) {
                            x_patch_start = (x_idxs[i] + -x_idxs[i] * dilations[i]) % dilations[i];
                        }

                        if (x_patch_end > feature_shape[i]) {
                            x_patch_end = feature_shape[i];
                        }

                        connx_Slice_set(&x_patch_slices[i], x_patch_start, x_patch_end, x_patch_step);

                        int32_t x_padded_start = x_idxs[i] < 0 ? -x_idxs[i] : 0;
                        k_idx[i] = x_padded_start;
                    }

                    connx_Iterator x_patch_iter;
                    connx_Iterator_init(&x_patch_iter, feature_dim, x_patch_slices);

                    // max pool
                    int32_t x_patch_batch = connx_Iterator_get_batch_size(&x_patch_iter, feature_shape);
                    connx_Iterator_rewind(&x_patch_iter, x_patch_batch);

                    int32_t kernel_offset = -1;
                    if (x_patch_batch == 1) {
                        while (connx_Iterator_next(&x_patch_iter, x_patch_batch)) {
                            int32_t x_patch_offset = connx_Iterator_offset(&x_patch_iter, feature_shape);
                            // clang-format off
                            {{TYPE}} tmp_y = X_flatten[x_patch_offset];
                            // clang-format on
                            if (kernel_offset < 0 || tmp_y > y) {
                                y = tmp_y;
                                kernel_offset = X_offset + x_patch_iter.idx;
                            }
                        }
                    } else {
                        while (connx_Iterator_next(&x_patch_iter, x_patch_batch)) {
                            int32_t x_patch_offset = connx_Iterator_offset(&x_patch_iter, feature_shape);
                            // clang-format off
                            {{TYPE}} tmp_y = CONNX_{{DTYPE}}_MIN;
                            // clang-format on
                            for (int i = 0; i < x_patch_batch; i++) {
                                if (X_flatten[x_patch_offset + i] > tmp_y) {
                                    tmp_y = X_flatten[x_patch_offset + i];
                                }
                            }
                            // clang-format off
                            int32_t tmp_kernel_offset = connx_{{DTYPE | to_name }}_argmax(x_patch_batch, &tmp_y, X_flatten + x_patch_offset);
                            // clang-format on

                            if (kernel_offset < 0 || tmp_y > y) {
                                y = tmp_y;
                                kernel_offset = X_offset + x_patch_iter.idx + tmp_kernel_offset;
                            }
                        }
                    }

                    // Convert offset to index
                    for (int32_t i = 1; i < feature_dim; i++) {
                        int32_t unit = connx_Int32_product(feature_dim - i, x_patch_iter.subshape + i);
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
                    Y_flatten[y_idx] = y;

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

    connx_Graph_set(graph, outputs[0], Y);
    if (output_count > 1) {
        connx_Graph_set(graph, outputs[1], Indices);
    }

    return CONNX_OK;
}
