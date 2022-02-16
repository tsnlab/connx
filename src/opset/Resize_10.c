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
#include <assert.h>
#include <string.h>

#include <connx/accel.h>
#include <connx/connx.h>

/*{% set supported_data_types = [
    INT8, INT16, INT32, INT64,
    UINT8, UINT16, UINT32, UINT64,
    FLOAT32, FLOAT64,
    ] %}*/
// TODO: STRING

/*{% set supported_modes = [
    'nearest', 'linear'
] %}*/

enum MODE {
    /*{% for mode in supported_modes %}*/
    {{mode | upper}},
    /*{% endfor %}*/
};

static enum MODE get_mode(const char* mode) {
    /*{% for _mode in supported_modes %}*/
    if (strncmp(mode, "{{_mode}}", {{_mode | length + 1}}) == 0) {
        return {{_mode | upper}};
    }
    /*{% endfor %}*/

    return -1;
}

/*
static int32_t _round_floor(float32_t f) {
    if (f - (int32_t)f > 0.5) {
        return (int32_t)f + 1;
    } else {
        return (int32_t)f;
    }
}

static int32_t _floor(float32_t f) {
    return (int32_t)f;
}

static int32_t _round_ceil(float32_t f) {
    return (int32_t)(f + 0.5);
}
*/

/*
static int32_t _ceil(float32_t f) {
    if (f >= 0) {
        int32_t i = (int32_t)f;
        if (i == f) {
            return i;
        } else {
            return i + 1;
        }
    } else {
        return (int32_t)f;
    }
}
*/

#define bound(v, size) ((v) < 0 ? 0 : (v) >= (size) ? (size) - 1 : (v))

/*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
static void interpolate_2d_nearest_{{DTYPE}}({{TYPE}}* Y_array, int32_t* X_shape, {{TYPE}}* X_array, int32_t* Y_shape, float32_t* scales, int32_t loop_count) {

    float32_t bases[2];
    float32_t steps[2];

    for (int32_t i = 0; i < 2; i++) {
        bases[i] = 0.5 / scales[i] - 0.5;
        steps[i] = 1 / scales[i];
    }

    int32_t X_step = connx_Int32_product(2, X_shape);

    for (int32_t loop = 0; loop < loop_count; loop++) {
        float32_t y0_idx = bases[0];
        for (int32_t y0 = 0; y0 < Y_shape[0]; y0++) {

            float32_t y1_idx = bases[1];
            for (int32_t y1 = 0; y1 < Y_shape[1]; y1++) {

                int32_t rounded_y0_idx = (int32_t)(y0_idx + 0.5);
                int32_t rounded_y1_idx = (int32_t)(y1_idx + 0.5);
                int32_t bounded_y0_idx = bound(rounded_y0_idx, X_shape[0]);
                int32_t bounded_y1_idx = bound(rounded_y1_idx, X_shape[1]);
                int32_t X_offset = bounded_y0_idx * X_shape[1] + bounded_y1_idx;

                *Y_array++ = X_array[X_offset];

                y1_idx += steps[1];
            }

            y0_idx += steps[0];
        }

        X_array += X_step;
    }
}
/*{% endfor %}*/

/*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
static void interpolate_2d_linear_{{DTYPE}}({{TYPE}}* Y_array, int32_t* X_shape, {{TYPE}}* X_array, int32_t* Y_shape, float32_t* scales, int32_t loop_count) {

    float32_t bases[2];
    float32_t steps[2];

    for (int32_t i = 0; i < 2; i++) {
        bases[i] = 0;
        steps[i] = 1 / scales[i];
    }

    int32_t X_step = connx_Int32_product(2, X_shape);

    for (int32_t loop = 0; loop < loop_count; loop++) {
        float32_t y0_idx = bases[0];
        for (int32_t y0 = 0; y0 < Y_shape[0]; y0++) {

            float32_t y1_idx = bases[1];
            for (int32_t y1 = 0; y1 < Y_shape[1]; y1++) {

                int32_t y0_idx_int = y0_idx >= 0 ? (int32_t)y0_idx : (int32_t)y0_idx - 1;
                int32_t y1_idx_int = y1_idx >= 0 ? (int32_t)y1_idx : (int32_t)y1_idx - 1;
                float32_t ratio_y0 = y0_idx - y0_idx_int;
                float32_t ratio_y1 = y1_idx - y1_idx_int;

                // coeffects
                float32_t coeffects[4] = {
                    (1 - ratio_y0) * (1 - ratio_y1), (1 - ratio_y0) * ratio_y1,
                    ratio_y0 * (1 - ratio_y1), ratio_y0 * ratio_y1
                };

                float32_t values[4] = {
                        X_array[bound(y0_idx_int, X_shape[0]) * X_shape[1] + bound(y1_idx_int, X_shape[1])],
                        X_array[bound(y0_idx_int, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 1, X_shape[1])],
                        X_array[bound(y0_idx_int + 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int, X_shape[1])],
                        X_array[bound(y0_idx_int + 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 1, X_shape[1])]
                    };

                *Y_array++ = ({{TYPE}})(coeffects[0] * values[0] + coeffects[1] * values[1] + 
                                        coeffects[2] * values[2] + coeffects[3] * values[3]);

                y1_idx += steps[1];
            }

            y0_idx += steps[0];
        }

        X_array += X_step;
    }
}
/*{% endfor %}*/

int Resize_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
        __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
        __attribute__((unused)) uint32_t attribute_count, void** attributes) {
    // Inputs
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]); // T
    connx_Tensor* scales = connx_Graph_get(graph, inputs[1]); // float32

    // Attributes
    enum MODE mode = get_mode(attributes[0]);
    if (mode < 0) {
        connx_error("mode '%s' is not supported yet", (char*)attributes[4]);
        return CONNX_NOT_SUPPORTED_ATTRIBUTE;
    }


    // prepare Y
    assert(scales->ndim == 1);
    assert(X->ndim == scales->shape[0]);

    int32_t shape[X->ndim];
    for (int i = 0; i < X->ndim; i++) {
        shape[i] = X->shape[i] * ((float32_t*)scales->buffer)[i];
    }

    connx_Tensor* Y = connx_Tensor_alloc(X->dtype, X->ndim, shape);
    if (Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], Y);

    // interpolate
    int32_t loop_count = connx_Int32_product(2, Y->shape);

    switch (X->ndim - 2) {
    case 2: // 2d interpolation
        switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
            case {{DTYPE}}: {
                {{TYPE}}* Y_buffer = ({{TYPE}}*)Y->buffer;
                {{TYPE}}* X_buffer = ({{TYPE}}*)X->buffer;

                switch (mode) {
                    case NEAREST:
                        interpolate_2d_nearest_{{DTYPE}}(Y_buffer, X->shape + 2, X_buffer, Y->shape + 2, &((float32_t*)scales->buffer)[2], loop_count);

                        break;
                    case LINEAR:
                        interpolate_2d_linear_{{DTYPE}}(Y_buffer, X->shape + 2, X_buffer, Y->shape + 2, &((float32_t*)scales->buffer)[2], loop_count);
                        break;
                    default:
                        assert(false); // Not possible
                }
           } 
                break;
        /*{% endfor %}*/
            default:
                connx_error("Not supported dtype: %d", X->dtype);
                return CONNX_NOT_SUPPORTED_FEATURE;
        }
        break;
    default:
        connx_error("Not supported Resize dimension: %u", X->ndim - 2);
        return CONNX_NOT_SUPPORTED_FEATURE;
    }

    return CONNX_OK;
}
