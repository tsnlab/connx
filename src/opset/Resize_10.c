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

/*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
static void interpolate_1d_nearest_{{DTYPE}}({{TYPE}}* Y_buffer, {{TYPE}}* X_buffer, int32_t shape, float32_t scale) {
    float32_t unit_scale = 1.0 / scale;
    float32_t base = unit_scale / 2.0;

    float32_t X_index = base;
    for (int32_t i = 0; i < shape; i++) {
        X_index += unit_scale;
        Y_buffer[i] = X_buffer[(int32_t)X_index];
        //Y_buffer[i] = 0;
    }
}
/*{% endfor %}*/

/*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
static void interpolate_1d_linear_{{DTYPE}}(__attribute__((unused)) {{TYPE}}* Y_buffer, __attribute__((unused)) {{TYPE}}* X_buffer, __attribute__((unused)) int32_t shape, __attribute__((unused)) float32_t scale) {
}
/*{% endfor %}*/

int Resize_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
        __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
        __attribute__((unused)) uint32_t attribute_count, void** attributes) {
    // Inputs
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]); // T
    connx_Tensor* scales = connx_Graph_get(graph, inputs[1]); // float32

    // Attributes
    const char* attr_mode = (const char*)attributes[0];

    // parse mode
    enum MODE mode;

    /*{% for mode in supported_modes %}*/
    if (strncmp(attr_mode, "{{mode}}", {{mode | length + 1}}) == 0) {
        mode = {{mode | upper}};
    } else
    /*{% endfor %}*/
    {
        connx_error("mode '%s' is not supported yet", attr_mode);
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
    int32_t Y_unit = connx_Int32_product(Y->ndim - 2, Y->shape + 2);
    int32_t X_unit = connx_Int32_product(X->ndim - 2, X->shape + 2);
    int32_t count = connx_Int32_product(2, Y->shape);
    float32_t* scales_array = (float32_t*)scales->buffer;

    if (X->ndim - 2 == 1) { // 1d interpolation
        switch (X->dtype) {
        /*{% for DTYPE, TYPE in loop_types(*supported_data_types) %}*/
            case {{DTYPE}}: {
                {{TYPE}}* Y_buffer = ({{TYPE}}*)Y->buffer;
                {{TYPE}}* X_buffer = ({{TYPE}}*)X->buffer;

                switch (mode) {
                    case NEAREST:
                        for (int32_t i = 0; i < count; i++) {
                            interpolate_1d_nearest_{{DTYPE}}(Y_buffer, X_buffer, X->shape[2 + 0], scales_array[0]);

                            Y_buffer += Y_unit;
                            X_buffer += X_unit;
                        }
                        break;
                    case LINEAR:
                        for (int32_t i = 0; i < count; i++) {
                            interpolate_1d_linear_{{DTYPE}}(Y_buffer, X_buffer, X->shape[2 + 0], scales_array[0]);

                            Y_buffer += Y_unit;
                            X_buffer += X_unit;
                        }
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
    } else if (X->ndim - 2 == 2) { // 2d interpolation
    } else if (X->ndim - 2 == 3) { // 2d interpolation
    } else {
        connx_error("Not supported Resize dimension: %u", X->ndim - 2);
        return CONNX_NOT_SUPPORTED_FEATURE;
    }

    return CONNX_OK;
}
