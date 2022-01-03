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
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <connx/accel.h>
#include <connx/connx.h>

/*{% set supported_data_types = [
    INT8, INT16, INT32, INT64,
    UINT8, UINT16, UINT32, UINT64,
    FLOAT32, FLOAT64,
    BOOL,
    ] %}*/
// TODO: STRING

/*{% set supported_roi_types = [ FLOAT32, FLOAT64, ] %}*/

/*{% set supported_transform_modes = [
    'half_pixel',
    'pytorch_half_pixel',
    'align_corners',
    'asymmetric',
    'tf_crop_and_resize',
] %}*/

/*{% set supported_modes = [
    'nearest', 'linear', 'cubic',
] %}*/

/*{% set supported_nearest_modes = [
    'round_prefer_floor', 'floor',
    'round_prefer_ceil', 'ceil',
] %}*/

enum TRANSFORM_MODE {
    /*{% for mode in supported_transform_modes %}*/
    {{mode | upper}},
    /*{% endfor %}*/
};

enum MODE {
    /*{% for mode in supported_modes %}*/
    {{mode | upper}},
    /*{% endfor %}*/
};

enum NEAREST_MODE {
    /*{% for mode in supported_nearest_modes %}*/
    {{mode | upper}},
    /*{% endfor %}*/
};

static int Resize_prepare(connx_Tensor* X, connx_Tensor* roi, connx_Tensor* scales, connx_Tensor* sizes, enum TRANSFORM_MODE coordinate_transformation_mode, float32_t cubic_coeff_a, bool exclude_outside,
                          float32_t extrapolation_value, enum MODE mode, enum NEAREST_MODE nearest_mode, connx_Tensor** Y);

static int Resize_exec(connx_Tensor* X, connx_Tensor* Y, connx_Tensor* roi, connx_Tensor* scales, connx_Tensor* sizes,
                       enum TRANSFORM_MODE coordinate_transformation_mode, float32_t cubic_coeff_a, bool exclude_outside,
                       float32_t extrapolation_value, enum MODE mode, enum NEAREST_MODE nearest_mode);

static float interpolate_nd_float32(uint32_t* idxs, float* data, int32_t* shape, float* scales, uint32_t dim,
                                    float* roi, enum TRANSFORM_MODE coordinate_transformation_mode, float cubic_coeff_a,
                                    bool exclude_outside, float extrapolation_value, enum MODE mode,
                                    enum NEAREST_MODE nearest_mode);

static float interpolate_1d_float32(uint32_t idx, float* data, int32_t shape, float scale, float* roi,
                                    enum TRANSFORM_MODE coordinate_transformation_mode, float cubic_coeff_a, bool exclude_outside,
                                    float extrapolation_value, enum MODE mode, enum NEAREST_MODE nearest_mode);

int Resize(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs, uint32_t input_count,
           uint32_t* inputs, void** attributes) {
    // Inputs
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]); // T1
    // Optional Inputs
    connx_Tensor* roi = (input_count > 1) ? connx_Graph_get(graph, inputs[1]) : NULL;    // T2
    connx_Tensor* scales = (input_count > 2) ? connx_Graph_get(graph, inputs[2]) : NULL; // float32
    connx_Tensor* sizes = (input_count > 3) ? connx_Graph_get(graph, inputs[3]) : NULL;  // int64

    // One of the following must exclusively set
    assert((scales != NULL && sizes == NULL) || (scales == NULL && sizes != NULL));

    // Attributes
    const char* coordinate_transformation_mode = (const char*)attributes[0];
    const float32_t cubic_coeff_a = *(float32_t*)attributes[1];
    const int32_t exclude_outside = *(int32_t*)attributes[2];
    const float32_t extrapolation_value = *(float32_t*)attributes[3];
    const char* mode_ = (const char*)attributes[4];
    const char* nearest_mode_ = (const char*)attributes[5];

    connx_Tensor* Y = NULL;

    enum TRANSFORM_MODE transform_mode;
    enum MODE mode;
    enum NEAREST_MODE nearest_mode;

    // Check transform modes
    /*{% for mode in supported_transform_modes %}*/
    if (strncmp(coordinate_transformation_mode, "{{mode}}", {{mode | length + 1}}) == 0) {
        transform_mode = {{mode | upper}};
    } else
    /*{% endfor %}*/
    {
        return CONNX_NOT_SUPPORTED_ATTRIBUTE;
    }

    // Check modes
    /*{% for mode in supported_modes %}*/
    if (strncmp(mode_, "{{mode}}", {{mode | length + 1}}) == 0) {
        mode = {{mode | upper}};
    } else
    /*{% endfor %}*/
    {
        return CONNX_NOT_SUPPORTED_ATTRIBUTE;
    }

    // Check nearest modes
    /*{% for mode in supported_nearest_modes %}*/
    if (strncmp(nearest_mode_, "{{mode}}", {{mode | length + 1}}) == 0) {
        nearest_mode = {{mode | upper}};
    } else
    /*{% endfor %}*/
    {
        return CONNX_NOT_SUPPORTED_ATTRIBUTE;
    }

    int result;
    result = Resize_prepare(X, roi, scales, sizes, transform_mode, cubic_coeff_a, exclude_outside,
                                extrapolation_value, mode, nearest_mode, &Y);

    if (result != CONNX_OK) {
        if (Y != NULL) {
            connx_free(Y);
        }
        return result;
    }
    connx_Graph_set(graph, outputs[0], Y);

    result = Resize_exec(X, Y, roi, scales, sizes, transform_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);

    // TODO: Free unfreed memory.

    return result;
}

static int Resize_prepare(connx_Tensor* X, connx_Tensor* roi, connx_Tensor* scales, connx_Tensor* sizes, enum TRANSFORM_MODE coordinate_transformation_mode, __attribute__((unused))float32_t cubic_coeff_a, __attribute__((unused))bool exclude_outside,
                          __attribute__((unused))float32_t extrapolation_value, __attribute__((unused))enum MODE mode, __attribute__((unused))enum NEAREST_MODE nearest_mode, connx_Tensor** Y) {

    int32_t output_shape[X->ndim];

    // Calculate output shape
    if (sizes != NULL) {
        int64_t* sizes_array = (int64_t*)sizes->buffer;
        for (int32_t i = 0; i < X->ndim; i++) {
            output_shape[i] = (int32_t)sizes_array[i];
        }
        // Make scales from sizes
        if (scales == NULL) {
            scales = connx_Tensor_alloc(X->dtype, X->ndim, output_shape);
            // TODO: Free after use
            if (scales == NULL) {
                return CONNX_NOT_ENOUGH_MEMORY;
            }
            float32_t* scales_array = (float32_t*)scales->buffer;
            for (int32_t i = 0; i < X->ndim; i++) {
                scales_array[i] = (float32_t)sizes_array[i] / (float32_t)X->shape[i];
            }
        }
    } else {
        float32_t* scales_array = (float32_t*)scales->buffer;
        for (int32_t i = 0; i < X->ndim; i++) {
            output_shape[i] = (int32_t)(X->shape[i] * scales_array[i]);
        }
        // Make sizes from scales
        if (sizes == NULL) {
            sizes = connx_Tensor_alloc(CONNX_INT64, X->ndim, output_shape);
            // TODO: Free after use
            if (sizes == NULL) {
                return CONNX_NOT_ENOUGH_MEMORY;
            }
            int64_t* sizes_array = (int64_t*)sizes->buffer;
            for (int32_t i = 0; i < X->ndim; i++) {
                sizes_array[i] = (int64_t)output_shape[i];
            }
        }
    }


    switch (coordinate_transformation_mode) {
        /*{% for mode in supported_transform_modes %}*/
        case {{mode | upper}}: {
        /*{% if mode == 'tf_crop_and_resize' %}*/
        if (roi != NULL && roi->ndim != 1) {
            connx_error("roi must be 1D tensor");
            return CONNX_TENSOR_SHAPE_NOT_MATCHING;
        }

        __attribute__((unused)) float32_t new_roi[roi->shape[0]];
        // Reformat roi. [start1, ..., startN, end1, ..., endN] -> [start1, end1, ..., startN, endN]
        switch (roi->dtype) {
            /*{% for DTYPE, TYPE in loop_types(*supported_roi_types) %}*/
        case {{ DTYPE }}: {
            {{TYPE}}* roi_array = ({{TYPE}}*)roi->buffer;
            const int32_t len = roi->shape[0] / 2;
            for (int32_t i = 0; i < len; i++) {
                new_roi[i * 2] = roi_array[i];
                new_roi[i * 2 + 1] = roi_array[i + len];
            }
        } break;
            /*{% endfor %}*/ // roi->dtype
        default:
            connx_error("Unsupported ROI type: %d", roi->dtype);
            return CONNX_NOT_SUPPORTED_DATATYPE;
        }

        /*{% elif mode == 'half_pixel' %}*/
        // Convert roi to float32
        switch (roi->dtype) {
            /*{% for DTYPE, TYPE in loop_types(*supported_roi_types) %}*/
        case {{ DTYPE }}: {
            /*{% if DTYPE == 'FLOAT64' %}*/
            connx_Tensor* new_roi = connx_Tensor_alloc(FLOAT32, roi->ndim, roi->shape);
            float64_t* roi_array = (float64_t*)roi->buffer;
            float32_t* new_roi_array = (float32_t*)new_roi->buffer;
            for (int32_t i = 0; i < roi->shape[0]; i++) {
                new_roi_array[i] = (float32_t)roi_array[i];
            }

            // connx_free(roi);
            roi = new_roi;
            /*{% endif %}*/

        } break;
            /*{% endfor %}*/ // roi->dtype
        default: // roi->dtype
            connx_error("Unsupported ROI type: %d", roi->dtype);
            return CONNX_NOT_SUPPORTED_DATATYPE;
        }
        /*{% else %}*/
        // Do nothing
        /*{% endif %}*/ // mode == 'half_pixel'
        } break;
        /*{% endfor %}*/
        default: // transform mode
            connx_error("Unsupported coordinate transformation mode: %d", coordinate_transformation_mode);
            return CONNX_NOT_SUPPORTED_ATTRIBUTE;
    }

    *Y = connx_Tensor_alloc(X->dtype, X->ndim, output_shape);
    if (*Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    // All checks are fine
    return CONNX_OK;

}

static int Resize_exec(connx_Tensor* X, connx_Tensor* Y, connx_Tensor* roi, connx_Tensor* scales, connx_Tensor* sizes,
                       enum TRANSFORM_MODE coordinate_transformation_mode, float32_t cubic_coeff_a, bool exclude_outside,
                       float32_t extrapolation_value, enum MODE mode, enum NEAREST_MODE nearest_mode) {

    uint32_t dimension = Y->ndim;
    uint32_t shape[dimension];
    uint32_t idxs[dimension];

    bool has_next = false;
    int64_t* sizes_array = (int64_t*)sizes->buffer;
    for (uint32_t i = 0; i < dimension; i++) {
        idxs[i] = 0;
        shape[i] = sizes_array[i];

        if (shape[i] > 0) {
            has_next = true;
        }
    }

    if (!has_next) {
        return true;
    }

    // fixed type variables
    float* scales_array = (float*)scales->buffer;
    float* roi_array = (float*)roi->buffer;

    // non-fixed type variables
    float* Y_array = (float*)Y->buffer;
    float* X_array = (float*)X->buffer;

    int32_t total_count = connx_Int32_product(Y->ndim, Y->shape);

    for (int32_t i = 0; i < total_count; i++) {
        // process
        Y_array[i] =
            interpolate_nd_float32(idxs, X_array, X->shape, scales_array, X->ndim, roi_array, coordinate_transformation_mode,
                                   cubic_coeff_a, !!exclude_outside, extrapolation_value, mode, nearest_mode);

        // next index
        for (int32_t dim = dimension - 1; dim >= 0; dim--) {
            if (++idxs[dim] >= shape[dim]) {
                idxs[dim] = 0;
            } else {
                break;
            }
        }
    }

    return CONNX_OK;
}

static float interpolate_nd_float32(uint32_t* idxs, float* data, int32_t* shape, float* scales, uint32_t dim,
                                    float* roi, enum TRANSFORM_MODE coordinate_transformation_mode, float cubic_coeff_a,
                                    bool exclude_outside, float extrapolation_value, enum MODE mode,
                                    enum NEAREST_MODE nearest_mode) {
    if (dim == 1) {
        return interpolate_1d_float32(idxs[0], data, shape[0], scales[0], roi, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
    }

    uint32_t unit = 1;
    for (uint32_t i = 0; i < dim; i++) {
        unit *= shape[dim - i - 1];
    }

    float interpolated[shape[0]];
    for (int32_t i = 0; i < shape[0]; i++) {
        interpolated[i] = interpolate_nd_float32(idxs + 1, data + unit * i, shape + 1, scales + 1, dim - 1, roi + 2,
                                                 coordinate_transformation_mode, cubic_coeff_a, exclude_outside,
                                                 extrapolation_value, mode, nearest_mode);
    }

    return interpolate_1d_float32(idxs[0], interpolated, shape[0], scales[0], roi, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
}

static float interpolate_1d_float32(uint32_t idx, float* data, int32_t shape, float scale, float* roi,
                                    enum TRANSFORM_MODE coordinate_transformation_mode, float cubic_coeff_a, bool exclude_outside,
                                    float extrapolation_value, enum MODE mode, enum NEAREST_MODE nearest_mode) {
    float origin_index = 0;
    float output_shape = scale * shape;

    // Get original coordinate
    switch (coordinate_transformation_mode) {
        case HALF_PIXEL:
            origin_index = ((float)idx + 0.5) / scale - 0.5;
            break;
        case PYTORCH_HALF_PIXEL:
            origin_index = output_shape > 1 ? ((float)idx + 0.5) / scale - 0.5 : 0;
            break;
        case ALIGN_CORNERS:
            origin_index = (float)idx * (shape - 1) / (output_shape - 1);
            break;
        case ASYMMETRIC:
            origin_index = idx / scale;
            break;
        case TF_CROP_AND_RESIZE:
            origin_index = output_shape > 1
                               ? roi[0] * (shape - 1) + idx * (roi[1] - roi[0]) * (shape - 1) / (output_shape - 1)
                               : 0.5 * (roi[0] + roi[1]) * (shape - 1);

            if (origin_index < 0 || origin_index > shape - 1) {
                return extrapolation_value;
            }

            break;
        default:
            abort();
    }

    int32_t origin_index_int = origin_index >= 0 ? (int32_t)origin_index : (int32_t)origin_index - 1;
    float ratio = 0;
    if (origin_index == origin_index_int) {
        ratio = 1;
    } else {
        ratio = origin_index - origin_index_int;
    }

    // Get coeffects
    float coeffects[4];
    uint32_t coeffects_count = 0;
    switch (mode) {
    case NEAREST:
        if (ratio == (int32_t)ratio) {
            coeffects[0] = 0;
            coeffects[1] = 1;
            coeffects_count = 2;
        } else {
            switch (nearest_mode) {
                case ROUND_PREFER_FLOOR:
                    if (ratio <= 0.5) {
                        coeffects[0] = 1;
                        coeffects[1] = 0;
                    } else {
                        coeffects[0] = 0;
                        coeffects[1] = 1;
                    }
                    coeffects_count = 2;
                    break;
                case ROUND_PREFER_CEIL:
                    if (ratio < 0.5) {
                        coeffects[0] = 1;
                        coeffects[1] = 0;
                    } else {
                        coeffects[0] = 0;
                        coeffects[1] = 1;
                    }
                    coeffects_count = 2;
                    break;
                case FLOOR:
                    coeffects[0] = 1;
                    coeffects[1] = 0;
                    coeffects_count = 2;
                    break;
                case CEIL:
                    coeffects[0] = 0;
                    coeffects[1] = 1;
                    coeffects_count = 2;
                    break;
                default:
                    abort();
            }
        }
        break;
    case LINEAR:
        coeffects[0] = 1 - ratio;
        coeffects[1] = ratio;
        coeffects_count = 2;
        break;
    case CUBIC:
        coeffects[0] = ((cubic_coeff_a * (ratio + 1) - 5 * cubic_coeff_a) * (ratio + 1) + 8 * cubic_coeff_a) * (ratio + 1) - 4 * cubic_coeff_a;
        coeffects[0] =
            ((cubic_coeff_a * (ratio + 1) - 5 * cubic_coeff_a) * (ratio + 1) + 8 * cubic_coeff_a) * (ratio + 1) -
            4 * cubic_coeff_a;
        coeffects[1] = ((cubic_coeff_a + 2) * ratio - (cubic_coeff_a + 3)) * ratio * ratio + 1;
        coeffects[2] = ((cubic_coeff_a + 2) * (1 - ratio) - (cubic_coeff_a + 3)) * (1 - ratio) * (1 - ratio) + 1;
        coeffects[3] =
            ((cubic_coeff_a * ((1 - ratio) + 1) - 5 * cubic_coeff_a) * ((1 - ratio) + 1) + 8 * cubic_coeff_a) *
                ((1 - ratio) + 1) -
            4 * cubic_coeff_a;
        coeffects_count = 4;
        break;
    default:
        abort();
    }

    // Calculate base
    int32_t idx_base;
    if (origin_index == origin_index_int) {
        idx_base = origin_index_int - coeffects_count / 2;
    } else {
        idx_base = origin_index_int - coeffects_count / 2 + 1;
    }

    // exclude_outside
    if (exclude_outside) {
        float sum = 0;
        for (uint32_t i = 0; i < coeffects_count; i++) {
            int j = idx_base + i;
            if (j < 0 || j >= shape) {
                coeffects[i] = 0;
            } else {
                sum += coeffects[i];
            }
        }

        if (sum != 0) {
            for (uint32_t i = 0; i < coeffects_count; i++) {
                coeffects[i] /= sum;
            }
        }
    }

    float interpolate = 0;
    for (uint32_t i = 0; i < coeffects_count; i++) {
        float value;
        int j = idx_base + i;
        if (j < 0) { // left edge padding
            value = data[0];
        } else if (j >= shape) { // right edge padding
            value = data[shape - 1];
        } else {
            value = data[j];
        }

        interpolate += coeffects[i] * value;
    }

    return interpolate;
}
