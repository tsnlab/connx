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

#include <connx/accel.h>
#include <connx/connx.h>

TEMPLATE_START(FLOAT32, FLOAT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
#define connx_TEMPLATE_NAME_add connx_Float32_add
static void _conv_TEMPLATE_NAME(TEMPLATE_TYPE* Y_flatten, TEMPLATE_TYPE* X_flatten, int32_t feature_dim,
                                int32_t* feature_shape, connx_Iterator* x_iter, TEMPLATE_TYPE* W_flatten,
                                int32_t* kernel_shape, int32_t* dilations) {

    while (connx_Iterator_next(x_iter, 1)) {
        TEMPLATE_TYPE y = 0;

        // Calculate x_patch_slices and w_slices
        connx_Slice x_patch_slices[feature_dim];
        connx_Slice w_slices[feature_dim];

        int32_t x_idxs[feature_dim];
        connx_Iterator_indices(x_iter, x_idxs);

        for (int32_t i = 0; i < feature_dim; i++) {
            int32_t x_patch_start = x_idxs[i];
            int32_t x_patch_end = x_idxs[i] + kernel_shape[i] * dilations[i];
            int32_t x_patch_step = dilations[i];

            int32_t w_start = 0;
            int32_t w_end = kernel_shape[i];
            int32_t w_step = 1;

            if (x_patch_start < 0) {
                x_patch_start = (x_idxs[i] + -x_idxs[i] * dilations[i]) % dilations[i];
                w_start += (x_patch_start - x_idxs[i]) / dilations[i];
            }

            if (x_patch_end > feature_shape[i]) {
                int32_t end = x_patch_end;
                x_patch_end = feature_shape[i];
                w_end -= (end - x_patch_end) / dilations[i];
            }

            connx_Slice_set(&x_patch_slices[i], x_patch_start, x_patch_end, x_patch_step);
            connx_Slice_set(&w_slices[i], w_start, w_end, w_step);
        }

        // Convolution by iteration
        connx_Iterator x_patch_iter;
        connx_Iterator_init(&x_patch_iter, feature_dim, x_patch_slices);

        connx_Iterator w_iter;
        connx_Iterator_init(&w_iter, feature_dim, w_slices);

        int32_t x_patch_batch = connx_Iterator_get_batch_size(&x_patch_iter, feature_shape);
        int32_t w_batch = connx_Iterator_get_batch_size(&w_iter, kernel_shape);
        int32_t mini_batch = x_patch_batch < w_batch ? x_patch_batch : w_batch;

        connx_Iterator_rewind(&x_patch_iter, mini_batch);
        connx_Iterator_rewind(&w_iter, mini_batch);

        while (connx_Iterator_next(&x_patch_iter, mini_batch) && connx_Iterator_next(&w_iter, mini_batch)) {
            int32_t x_patch_offset = connx_Iterator_offset(&x_patch_iter, feature_shape);
            int32_t w_offset = connx_Iterator_offset(&w_iter, kernel_shape);

            y += connx_TEMPLATE_NAME_mul_and_sum(mini_batch, (TEMPLATE_TYPE*)X_flatten + x_patch_offset,
                                                 (TEMPLATE_TYPE*)W_flatten + w_offset);
        }

        *Y_flatten++ += y;
    }
}

struct Parameter_TEMPLATE_NAME {
    TEMPLATE_TYPE* Y_flatten;
    TEMPLATE_TYPE* X_flatten;
    TEMPLATE_TYPE* B_flatten;
    TEMPLATE_TYPE* W_flatten;
    int32_t feature_dim;
    int32_t* feature_shape;
    int32_t* kernel_shape;
    int32_t* dilations;
    connx_Iterator x_iter;
    int32_t X_unit;
    int32_t W_unit;
    int32_t Y_unit;
    int32_t channel_count;
};

static void* run_TEMPLATE_NAME(void* context) {
    struct Parameter_TEMPLATE_NAME* params = context;

    TEMPLATE_TYPE* Y_flatten = params->Y_flatten;
    TEMPLATE_TYPE* X_flatten = params->X_flatten;
    TEMPLATE_TYPE* B_flatten = params->B_flatten;
    TEMPLATE_TYPE* W_flatten = params->W_flatten;
    int32_t feature_dim = params->feature_dim;
    int32_t* feature_shape = params->feature_shape;
    int32_t* kernel_shape = params->kernel_shape;
    int32_t* dilations = params->dilations;
    connx_Iterator* x_iter = &params->x_iter;
    int32_t X_unit = params->X_unit;
    int32_t W_unit = params->W_unit;
    int32_t Y_unit = params->Y_unit;
    int32_t channel_count = params->channel_count;

    for (int32_t channel = 0; channel < channel_count; channel++) {
        _conv_TEMPLATE_NAME(Y_flatten, X_flatten, feature_dim, feature_shape, x_iter, W_flatten, kernel_shape,
                            dilations);

        X_flatten += X_unit;
        W_flatten += W_unit;
    }

    if (B_flatten != NULL) {
        TEMPLATE_TYPE B_array[Y_unit];
        connx_TEMPLATE_NAME_broadcast(Y_unit, B_array, 1, B_flatten);
        connx_TEMPLATE_NAME_add(Y_unit, Y_flatten, Y_flatten, B_array);
        B_flatten++;
    }

    return NULL;
}

TEMPLATE_END()

int Conv_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs, uint32_t input_count,
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
        TEMPLATE_START(FLOAT32, FLOAT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
#define connx_TEMPLATE_NAME_add connx_Float32_add
#define connx_TEMPLATE_NAME_broadcast connx_Float32_broadcast
    case TEMPLATE_DTYPE: {
        TEMPLATE_TYPE* X_flatten = (TEMPLATE_TYPE*)X->buffer;
        TEMPLATE_TYPE* Y_flatten = (TEMPLATE_TYPE*)Y->buffer;
        TEMPLATE_TYPE* B_flatten = NULL;

        int32_t batch_count = X->shape[0];
        int32_t channel_count = W->shape[1];
        int32_t feature_group = W->shape[0] / group;

        int32_t X_unit = connx_Int32_product(feature_dim, feature_shape);
        int32_t W_unit = connx_Int32_product(W->ndim - 2, W->shape + 2);
        int32_t W_feature_unit = W->shape[1] * W_unit;
        int32_t Y_unit = connx_Int32_product(feature_dim, output_shape);

        int32_t work_count = batch_count * group * feature_group;
        struct Parameter_TEMPLATE_NAME works[work_count];

        for (int32_t batch = 0, work_id = 0; batch < batch_count; batch++) {
            if (B != NULL) {
                B_flatten = (TEMPLATE_TYPE*)B->buffer;
            }

            TEMPLATE_TYPE* W_flatten = (TEMPLATE_TYPE*)W->buffer;

            for (int32_t g = 0, feature_map = 0; g < group; g++) {
                for (int32_t f = 0; f < feature_group; f++, feature_map++) {
                    works[work_id].Y_flatten = Y_flatten;
                    works[work_id].X_flatten = X_flatten;
                    works[work_id].B_flatten = B_flatten;
                    works[work_id].W_flatten = W_flatten;
                    works[work_id].feature_dim = feature_dim;
                    works[work_id].feature_shape = feature_shape;
                    works[work_id].kernel_shape = kernel_shape;
                    works[work_id].dilations = dilations;
                    works[work_id].x_iter = x_iter;
                    works[work_id].X_unit = X_unit;
                    works[work_id].W_unit = W_unit;
                    works[work_id].Y_unit = Y_unit;
                    works[work_id].channel_count = channel_count;
                    work_id++;

                    /*
                    for (int32_t channel = 0; channel < channel_count; channel++) {
                        _conv_TEMPLATE_NAME(Y_flatten, X_flatten + channel * X_unit, feature_dim, feature_shape,
                                            &x_iter, W_flatten + channel * W_unit, kernel_shape, dilations);
                    }

                    if (B_flatten != NULL) {
                        TEMPLATE_TYPE B_array[Y_unit];
                        connx_TEMPLATE_NAME_broadcast(Y_unit, B_array, 1, B_flatten);
                        connx_TEMPLATE_NAME_add(Y_unit, Y_flatten, Y_flatten, B_array);
                        B_flatten++;
                    }
                    */
                    if (B_flatten != NULL) {
                        B_flatten++;
                    }

                    Y_flatten += Y_unit;
                    W_flatten += W_feature_unit;
                }

                X_flatten += channel_count * X_unit;
            }
        }

        connx_Thread_run_all(run_TEMPLATE_NAME, work_count, works, sizeof(struct Parameter_TEMPLATE_NAME));

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
