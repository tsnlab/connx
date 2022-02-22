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

/*{% set supported_coordinate_transformation_modes = [
    'half_pixel',
    'pytorch_half_pixel',
    'align_corners',
    'asymmetric',
    'tf_half_pixel_for_nn',
] %}*/

enum COORDINATE_TRANSFORMATION_MODE {
    /*{% for mode in supported_coordinate_transformation_modes %}*/
    {{mode | upper}},
    /*{% endfor %}*/
};

/*{% set supported_modes = [
    'nearest', 'linear', 'cubic'
] %}*/

enum MODE {
    /*{% for mode in supported_modes %}*/
    {{mode | upper}},
    /*{% endfor %}*/
};

/*{% set supported_nearest_modes = [
    'round_prefer_floor', 'floor',
    'round_prefer_ceil', 'ceil',
] %}*/

enum NEAREST_MODE {
    /*{% for mode in supported_nearest_modes %}*/
    {{mode | upper}},
    /*{% endfor %}*/
};

static enum COORDINATE_TRANSFORMATION_MODE get_coordinate_transformation_mode(const char* mode) {
    /*{% for _mode in supported_coordinate_transformation_modes %}*/
    if (strncmp(mode, "{{_mode}}", {{_mode | length + 1}}) == 0) {
        return {{_mode | upper}};
    }
    /*{% endfor %}*/

    return -1;
}

static enum MODE get_mode(const char* mode) {
    /*{% for _mode in supported_modes %}*/
    if (strncmp(mode, "{{_mode}}", {{_mode | length + 1}}) == 0) {
        return {{_mode | upper}};
    }
    /*{% endfor %}*/

    return -1;
}

static enum NEAREST_MODE get_nearest_mode(const char* mode) {
    /*{% for _mode in supported_nearest_modes %}*/
    if (strncmp(mode, "{{_mode}}", {{_mode | length + 1}}) == 0) {
        return {{_mode | upper}};
    }
    /*{% endfor %}*/

    return -1;
}

static void calc_coord(float32_t* bases, float32_t* steps,
                       enum COORDINATE_TRANSFORMATION_MODE coordinate_transformation_mode, int32_t ndim,
                       int32_t* X_shape, float32_t* scales) {
    switch (coordinate_transformation_mode) {
    case HALF_PIXEL:
        for (int32_t i = 0; i < ndim; i++) {
            bases[i] = 0.5 / scales[i] - 0.5;
            steps[i] = 1 / scales[i];
        }
        break;
    case PYTORCH_HALF_PIXEL:
        for (int32_t i = 0; i < ndim; i++) {
            if (X_shape[i] * scales[i] > 1) {
                bases[i] = 0.5 / scales[i] - 0.5;
                steps[i] = 1 / scales[i];
            } else {
                bases[i] = 0;
                steps[i] = 0;
            }
        }
        break;
    case ALIGN_CORNERS:
        for (int32_t i = 0; i < ndim; i++) {
            bases[i] = 0;

            float32_t Y_length = X_shape[i] * scales[i];
            if (Y_length == 1) {
                steps[i] = 0;
            } else {
                steps[i] = (float32_t)(X_shape[i] - 1) / (float32_t)(Y_length - 1);
            }
        }
        break;
    case ASYMMETRIC:
        for (int32_t i = 0; i < ndim; i++) {
            bases[i] = 0;
            steps[i] = 1 / scales[i];
        }
        break;
    case TF_HALF_PIXEL_FOR_NN:
        for (int32_t i = 0; i < ndim; i++) {
            bases[i] = 0.5 / scales[i];
            steps[i] = 1 / scales[i];
        }
        break;
        /*
    case TF_CROP_AND_RESIZE:
        for (int32_t i = 0; i < ndim; i++) {
            float32_t Y_length = X_shape[i] * scales[i];
            float32_t start_x = roi[i];
            float32_t end_x = roi[i + ndim];
            if (Y_length > 1) {
                bases[i] = start_x * (X_shape[i] - 1);
                steps[i] = (end_x - start_x) * (X_shape[i] - 1) / (Y_length - 1);
            } else {
                bases[i] = 0.5 * (start_x + end_x) * (X_shape[i] - 1);
                steps[i] = 0;
            }
        }
        break;
        */
    default:
        assert(false);
        break;
    }
}

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

#define bound(v, size) ((v) < 0 ? 0 : (v) >= (size) ? (size)-1 : (v))

/*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
// clang-format off
static void interpolate_2d_nearest_{{DTYPE}}({{TYPE}} * Y_array, int32_t* X_shape, {{TYPE}} * X_array,
                                              // clang-format on
                                              int32_t* Y_shape, float32_t* bases, float32_t* steps,
                                              enum NEAREST_MODE nearest_mode, int32_t loop_count) {

    int32_t (*round)(float32_t);

    switch (nearest_mode) {
    case ROUND_PREFER_FLOOR:
        round = _round_floor;
        break;
    case FLOOR:
        round = _floor;
        break;
    case ROUND_PREFER_CEIL:
        round = _round_ceil;
        break;
    case CEIL:
        round = _ceil;
        break;
    default:
        // Not possible
        assert(false);
    }

    int32_t X_step = connx_Int32_product(2, X_shape);

    for (int32_t loop = 0; loop < loop_count; loop++) {
        float32_t y0_idx = bases[0];
        for (int32_t y0 = 0; y0 < Y_shape[0]; y0++) {

            float32_t y1_idx = bases[1];
            for (int32_t y1 = 0; y1 < Y_shape[1]; y1++) {

                int32_t rounded_y0_idx = round(y0_idx);
                int32_t rounded_y1_idx = round(y1_idx);
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
// clang-format off
static void interpolate_2d_linear_{{DTYPE}}({{TYPE}} * Y_array, int32_t* X_shape, {{TYPE}} * X_array, int32_t* Y_shape,
                                             // clang-format on
                                             float32_t* bases, float32_t* steps, bool exclude_outside,
                                             int32_t loop_count) {

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
                float32_t coeffects[4] = {(1 - ratio_y0) * (1 - ratio_y1), (1 - ratio_y0) * ratio_y1,
                                          ratio_y0 * (1 - ratio_y1), ratio_y0 * ratio_y1};

                // exclude outside
                if (exclude_outside) {
                    float32_t excluded = 0.0;
                    if (y0_idx_int < 0) {
                        excluded += coeffects[0];
                        excluded += coeffects[1];
                        coeffects[0] = 0;
                        coeffects[1] = 0;
                    }

                    if (y0_idx_int + 1 >= X_shape[0]) {
                        excluded += coeffects[2];
                        excluded += coeffects[3];
                        coeffects[2] = 0;
                        coeffects[3] = 0;
                    }

                    if (y1_idx_int < 0) {
                        excluded += coeffects[0];
                        excluded += coeffects[2];
                        coeffects[0] = 0;
                        coeffects[2] = 0;
                    }

                    if (y1_idx_int + 1 >= X_shape[1]) {
                        excluded += coeffects[1];
                        excluded += coeffects[3];
                        coeffects[1] = 0;
                        coeffects[3] = 0;
                    }

                    if (excluded != 0) {
                        float32_t included = 1 - excluded;
                        coeffects[0] /= included;
                        coeffects[1] /= included;
                        coeffects[2] /= included;
                        coeffects[3] /= included;
                    }
                }

                float32_t values[4] = {
                    X_array[bound(y0_idx_int, X_shape[0]) * X_shape[1] + bound(y1_idx_int, X_shape[1])],
                    X_array[bound(y0_idx_int, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 1, X_shape[1])],
                    X_array[bound(y0_idx_int + 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int, X_shape[1])],
                    X_array[bound(y0_idx_int + 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 1, X_shape[1])]};

                *Y_array++ = ({{TYPE}})(coeffects[0] * values[0] + coeffects[1] * values[1] + coeffects[2] * values[2] +
                                        coeffects[3] * values[3]);

                y1_idx += steps[1];
            }

            y0_idx += steps[0];
        }

        X_array += X_step;
    }
}
/*{% endfor %}*/

/*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
// clang-format off
static void interpolate_2d_cubic_{{DTYPE}}({{TYPE}} * Y_array, int32_t* X_shape, {{TYPE}} * X_array, int32_t* Y_shape,
                                            // clang-format on
                                            float32_t* bases, float32_t* steps, bool exclude_outside,
                                            float32_t cubic_coeff_a, int32_t loop_count) {

    int32_t X_step = connx_Int32_product(2, X_shape);

    for (int32_t loop = 0; loop < loop_count; loop++) {
        float32_t y0_idx = bases[0];
        for (int32_t y0 = 0; y0 < Y_shape[0]; y0++) {

            float32_t y1_idx = bases[1];
            for (int32_t y1 = 0; y1 < Y_shape[1]; y1++) {

                int32_t y0_idx_int = y0_idx >= 0 ? (int32_t)y0_idx : (int32_t)y0_idx - 1;
                int32_t y1_idx_int = y1_idx >= 0 ? (int32_t)y1_idx : (int32_t)y1_idx - 1;
                float32_t y0_ratio = y0_idx - y0_idx_int;
                float32_t y1_ratio = y1_idx - y1_idx_int;

                // coeffects
                float32_t y0_coeffects[4] = {
                    ((cubic_coeff_a * (y0_ratio + 1) - 5 * cubic_coeff_a) * (y0_ratio + 1) + 8 * cubic_coeff_a) *
                            (y0_ratio + 1) -
                        4 * cubic_coeff_a,
                    ((cubic_coeff_a + 2) * y0_ratio - (cubic_coeff_a + 3)) * y0_ratio * y0_ratio + 1,
                    ((cubic_coeff_a + 2) * (1 - y0_ratio) - (cubic_coeff_a + 3)) * (1 - y0_ratio) * (1 - y0_ratio) + 1,
                    ((cubic_coeff_a * ((1 - y0_ratio) + 1) - 5 * cubic_coeff_a) * ((1 - y0_ratio) + 1) +
                     8 * cubic_coeff_a) *
                            ((1 - y0_ratio) + 1) -
                        4 * cubic_coeff_a};

                float32_t y1_coeffects[4] = {
                    ((cubic_coeff_a * (y1_ratio + 1) - 5 * cubic_coeff_a) * (y1_ratio + 1) + 8 * cubic_coeff_a) *
                            (y1_ratio + 1) -
                        4 * cubic_coeff_a,
                    ((cubic_coeff_a + 2) * y1_ratio - (cubic_coeff_a + 3)) * y1_ratio * y1_ratio + 1,
                    ((cubic_coeff_a + 2) * (1 - y1_ratio) - (cubic_coeff_a + 3)) * (1 - y1_ratio) * (1 - y1_ratio) + 1,
                    ((cubic_coeff_a * ((1 - y1_ratio) + 1) - 5 * cubic_coeff_a) * ((1 - y1_ratio) + 1) +
                     8 * cubic_coeff_a) *
                            ((1 - y1_ratio) + 1) -
                        4 * cubic_coeff_a};

                float32_t coeffects[16];
                for (int32_t y0_co_idx = 0; y0_co_idx < 4; y0_co_idx++) {
                    float32_t y0_coeffect = y0_coeffects[y0_co_idx];

                    for (int32_t y1_co_idx = 0; y1_co_idx < 4; y1_co_idx++) {
                        float32_t y1_coeffect = y1_coeffects[y1_co_idx];

                        coeffects[y0_co_idx * 4 + y1_co_idx] = y0_coeffect * y1_coeffect;
                    }
                }

                // exclude outside
                if (exclude_outside) {
                    float32_t excluded = 0.0;
                    if (y0_idx_int < 0) {
                        excluded += coeffects[0 * 4 + 0];
                        excluded += coeffects[0 * 4 + 1];
                        excluded += coeffects[0 * 4 + 2];
                        excluded += coeffects[0 * 4 + 3];
                        excluded += coeffects[1 * 4 + 0];
                        excluded += coeffects[1 * 4 + 1];
                        excluded += coeffects[1 * 4 + 2];
                        excluded += coeffects[1 * 4 + 3];

                        coeffects[0 * 4 + 0] = 0;
                        coeffects[0 * 4 + 1] = 0;
                        coeffects[0 * 4 + 2] = 0;
                        coeffects[0 * 4 + 3] = 0;
                        coeffects[1 * 4 + 0] = 0;
                        coeffects[1 * 4 + 1] = 0;
                        coeffects[1 * 4 + 2] = 0;
                        coeffects[1 * 4 + 3] = 0;
                    } else if (y0_idx_int - 1 < 0) {
                        excluded += coeffects[0 * 4 + 0];
                        excluded += coeffects[0 * 4 + 1];
                        excluded += coeffects[0 * 4 + 2];
                        excluded += coeffects[0 * 4 + 3];

                        coeffects[0 * 4 + 0] = 0;
                        coeffects[0 * 4 + 1] = 0;
                        coeffects[0 * 4 + 2] = 0;
                        coeffects[0 * 4 + 3] = 0;
                    }

                    if (y0_idx_int + 1 >= X_shape[0]) {
                        excluded += coeffects[2 * 4 + 0];
                        excluded += coeffects[2 * 4 + 1];
                        excluded += coeffects[2 * 4 + 2];
                        excluded += coeffects[2 * 4 + 3];
                        excluded += coeffects[3 * 4 + 0];
                        excluded += coeffects[3 * 4 + 1];
                        excluded += coeffects[3 * 4 + 2];
                        excluded += coeffects[3 * 4 + 3];

                        coeffects[2 * 4 + 0] = 0;
                        coeffects[2 * 4 + 1] = 0;
                        coeffects[2 * 4 + 2] = 0;
                        coeffects[2 * 4 + 3] = 0;
                        coeffects[3 * 4 + 0] = 0;
                        coeffects[3 * 4 + 1] = 0;
                        coeffects[3 * 4 + 2] = 0;
                        coeffects[3 * 4 + 3] = 0;
                    } else if (y0_idx_int + 2 >= X_shape[0]) {
                        excluded += coeffects[3 * 4 + 0];
                        excluded += coeffects[3 * 4 + 1];
                        excluded += coeffects[3 * 4 + 2];
                        excluded += coeffects[3 * 4 + 3];

                        coeffects[3 * 4 + 0] = 0;
                        coeffects[3 * 4 + 1] = 0;
                        coeffects[3 * 4 + 2] = 0;
                        coeffects[3 * 4 + 3] = 0;
                    }

                    if (y1_idx_int < 0) {
                        excluded += coeffects[0 * 4 + 0];
                        excluded += coeffects[1 * 4 + 0];
                        excluded += coeffects[2 * 4 + 0];
                        excluded += coeffects[3 * 4 + 0];
                        excluded += coeffects[0 * 4 + 1];
                        excluded += coeffects[1 * 4 + 1];
                        excluded += coeffects[2 * 4 + 1];
                        excluded += coeffects[3 * 4 + 1];

                        coeffects[0 * 4 + 0] = 0;
                        coeffects[1 * 4 + 0] = 0;
                        coeffects[2 * 4 + 0] = 0;
                        coeffects[3 * 4 + 0] = 0;
                        coeffects[0 * 4 + 1] = 0;
                        coeffects[1 * 4 + 1] = 0;
                        coeffects[2 * 4 + 1] = 0;
                        coeffects[3 * 4 + 1] = 0;
                    } else if (y1_idx_int - 1 < 0) {
                        excluded += coeffects[0 * 4 + 0];
                        excluded += coeffects[1 * 4 + 0];
                        excluded += coeffects[2 * 4 + 0];
                        excluded += coeffects[3 * 4 + 0];

                        coeffects[0 * 4 + 0] = 0;
                        coeffects[1 * 4 + 0] = 0;
                        coeffects[2 * 4 + 0] = 0;
                        coeffects[3 * 4 + 0] = 0;
                    }

                    if (y1_idx_int + 1 >= X_shape[1]) {
                        excluded += coeffects[0 * 4 + 2];
                        excluded += coeffects[1 * 4 + 2];
                        excluded += coeffects[2 * 4 + 2];
                        excluded += coeffects[3 * 4 + 2];
                        excluded += coeffects[0 * 4 + 3];
                        excluded += coeffects[1 * 4 + 3];
                        excluded += coeffects[2 * 4 + 3];
                        excluded += coeffects[3 * 4 + 3];

                        coeffects[0 * 4 + 2] = 0;
                        coeffects[1 * 4 + 2] = 0;
                        coeffects[2 * 4 + 2] = 0;
                        coeffects[3 * 4 + 2] = 0;
                        coeffects[0 * 4 + 3] = 0;
                        coeffects[1 * 4 + 3] = 0;
                        coeffects[2 * 4 + 3] = 0;
                        coeffects[3 * 4 + 3] = 0;
                    } else if (y1_idx_int + 2 >= X_shape[1]) {
                        excluded += coeffects[0 * 4 + 3];
                        excluded += coeffects[1 * 4 + 3];
                        excluded += coeffects[2 * 4 + 3];
                        excluded += coeffects[3 * 4 + 3];

                        coeffects[0 * 4 + 3] = 0;
                        coeffects[1 * 4 + 3] = 0;
                        coeffects[2 * 4 + 3] = 0;
                        coeffects[3 * 4 + 3] = 0;
                    }

                    if (excluded != 0) {
                        float32_t included = 1 - excluded;
                        for (int32_t k = 0; k < 16; k++) {
                            coeffects[k] /= included;
                        }
                    }
                }

                float32_t values[16] = {
                    X_array[bound(y0_idx_int - 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int - 1, X_shape[1])],
                    X_array[bound(y0_idx_int - 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 0, X_shape[1])],
                    X_array[bound(y0_idx_int - 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 1, X_shape[1])],
                    X_array[bound(y0_idx_int - 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 2, X_shape[1])],

                    X_array[bound(y0_idx_int + 0, X_shape[0]) * X_shape[1] + bound(y1_idx_int - 1, X_shape[1])],
                    X_array[bound(y0_idx_int + 0, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 0, X_shape[1])],
                    X_array[bound(y0_idx_int + 0, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 1, X_shape[1])],
                    X_array[bound(y0_idx_int + 0, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 2, X_shape[1])],

                    X_array[bound(y0_idx_int + 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int - 1, X_shape[1])],
                    X_array[bound(y0_idx_int + 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 0, X_shape[1])],
                    X_array[bound(y0_idx_int + 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 1, X_shape[1])],
                    X_array[bound(y0_idx_int + 1, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 2, X_shape[1])],

                    X_array[bound(y0_idx_int + 2, X_shape[0]) * X_shape[1] + bound(y1_idx_int - 1, X_shape[1])],
                    X_array[bound(y0_idx_int + 2, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 0, X_shape[1])],
                    X_array[bound(y0_idx_int + 2, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 1, X_shape[1])],
                    X_array[bound(y0_idx_int + 2, X_shape[0]) * X_shape[1] + bound(y1_idx_int + 2, X_shape[1])],
                };

                float32_t v = 0;
                for (int32_t k = 0; k < 16; k++) {
                    v += coeffects[k] * values[k];
                }
                *Y_array++ = ({{TYPE}})v;

                y1_idx += steps[1];
            }

            y0_idx += steps[0];
        }

        X_array += X_step;
    }
}
/*{% endfor %}*/

// clang-format off
int Resize_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
                           // clang-format on
                           uint32_t input_count, uint32_t* inputs, __attribute__((unused)) uint32_t attribute_count,
                           void** attributes) {
    // Inputs
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);                               // T1
    __attribute__((unused)) connx_Tensor* _roi = connx_Graph_get(graph, inputs[1]);    // T2
    connx_Tensor* _scales = connx_Graph_get(graph, inputs[2]);                         // float32 (can be empty)
    connx_Tensor* _sizes = input_count > 3 ? connx_Graph_get(graph, inputs[3]) : NULL; // int64 (optional)

    // Attributes
    enum COORDINATE_TRANSFORMATION_MODE coordinate_transformation_mode =
        get_coordinate_transformation_mode(attributes[0]);
    if (coordinate_transformation_mode < 0) {
        connx_error("coordinate_transformation_mode '%s' is not supported yet", (char*)attributes[0]);
        return CONNX_NOT_SUPPORTED_ATTRIBUTE;
    }

    float32_t cubic_coeff_a = *(float32_t*)attributes[1];

    int32_t exclude_outside = *(int32_t*)attributes[2];

    __attribute__((unused)) float32_t extrapolation_value = *(float32_t*)attributes[3];

    enum MODE mode = get_mode(attributes[4]);
    if (mode < 0) {
        connx_error("mode '%s' is not supported yet", (char*)attributes[4]);
        return CONNX_NOT_SUPPORTED_ATTRIBUTE;
    }

    enum NEAREST_MODE nearest_mode = get_nearest_mode(attributes[5]);
    if (nearest_mode < 0) {
        connx_error("nearest_mode '%s' is not supported yet", (char*)attributes[5]);
        return CONNX_NOT_SUPPORTED_ATTRIBUTE;
    }

    // Normalize roi
    /*
    float32_t roi[X->ndim * 2];
    if (coordinate_transformation_mode == TF_CROP_AND_RESIZE) {
        if (_roi->dtype == CONNX_FLOAT32) {
            memcpy(roi, _roi->buffer, X->ndim * sizeof(float32_t));
        } else if (_roi->dtype == CONNX_FLOAT64) {
            for (int32_t i = 0; i < X->ndim; i++) {
                roi[i] = ((float64_t*)_roi->buffer)[i];
            }
        } else {
            assert(false);
        }
    }
    */

    // Normalize scales and sizes
    float32_t scales[X->ndim - 2];
    int32_t sizes[X->ndim - 2];

    // Important note: dim 0 and 1 is assumed batch and channel
    if (_scales != NULL && _scales->ndim == 1 && _scales->shape[0] == X->ndim) {
        assert(((float32_t*)_scales->buffer)[0] == 1 && ((float32_t*)_scales->buffer)[1] == 1);

        memcpy(scales, &((float32_t*)_scales->buffer)[2], sizeof(float32_t) * (X->ndim - 2));
        for (int32_t i = 0; i < _scales->shape[0] - 2; i++) {
            sizes[i] = scales[i] * X->shape[2 + i];
        }
    } else if (_sizes != NULL && _sizes->ndim == 1 && _sizes->shape[0] == X->ndim) {
        assert(((int64_t*)_sizes->buffer)[0] == X->shape[0] && ((int64_t*)_sizes->buffer)[1] == X->shape[1]);

        for (int32_t i = 0; i < _sizes->shape[0] - 2; i++) {
            sizes[i] = ((int64_t*)_sizes->buffer)[2 + i];
            scales[i] = (float32_t)sizes[i] / (float32_t)X->shape[2 + i];
        }
    } else {
        assert(false);
    }

    // prepare Y
    int32_t shape[X->ndim];
    shape[0] = X->shape[0];
    shape[1] = X->shape[1];
    memcpy(shape + 2, sizes, (X->ndim - 2) * sizeof(int32_t));

    connx_Tensor* Y = connx_Tensor_alloc(X->dtype, X->ndim, shape);
    if (Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], Y);

    // calculate coordinations
    float32_t bases[X->ndim - 2];
    float32_t steps[X->ndim - 2];

    calc_coord(bases, steps, coordinate_transformation_mode, X->ndim - 2, X->shape + 2, scales);

    // interpolate
    int32_t loop_count = connx_Int32_product(2, Y->shape);

    switch (X->ndim - 2) {
    case 2: // 2d interpolation
        switch (X->dtype) {
            /*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
        case {{ DTYPE }}: {
            {{TYPE}}* Y_buffer = ({{TYPE}}*)Y->buffer;
            {{TYPE}}* X_buffer = ({{TYPE}}*)X->buffer;

            switch (mode) {
            case NEAREST:
                // clang-format off
                interpolate_2d_nearest_{{DTYPE}}(Y_buffer, X->shape + 2, X_buffer, Y->shape + 2, bases, steps,
                                                  // clang-format on
                                                  nearest_mode, loop_count);

                break;
            case LINEAR:
                // clang-format off
                interpolate_2d_linear_{{DTYPE}}(Y_buffer, X->shape + 2, X_buffer, Y->shape + 2, bases, steps,
                                                 // clang-format on
                                                 exclude_outside == 1, loop_count);
                break;
            case CUBIC:
                // clang-format off
                interpolate_2d_cubic_{{DTYPE}}(Y_buffer, X->shape + 2, X_buffer, Y->shape + 2, bases, steps,
                                                // clang-format on
                                                exclude_outside == 1, cubic_coeff_a, loop_count);
                break;
            default:
                assert(false); // Not possible
            }
        } break;
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
