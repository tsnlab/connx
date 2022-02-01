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
#include <string.h>

#include <connx/accel.h>
#include <connx/connx.h>

static int32_t clip(int32_t value, int32_t min, int32_t max) {
    return value < min ? min : (value > max ? max : value);
}

static int32_t get_input_index(const int32_t ndim, const int32_t* input_shape, const int32_t* output_shape,
                        const int32_t* starts, __attribute__((unused)) const int32_t* steps,
                        const int32_t output_offset);

int Slice_{{op_version}}(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
          __attribute__((unused)) uint32_t input_count, uint32_t* inputs,
          __attribute__((unused)) uint32_t attribute_count, __attribute__((unused)) void** attributes) {
    /*{% set supported_data_types = [
        INT8, INT16, INT32, INT64,
        UINT8, UINT16, UINT32, UINT64,
        FLOAT32, FLOAT64,
        BOOL,
        ] %}*/
    // TODO: STRING

    /*{% set supported_index_types = [ INT32, INT64, ] %}*/
    connx_Tensor* data = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* starts = connx_Graph_get(graph, inputs[1]);
    connx_Tensor* ends = connx_Graph_get(graph, inputs[2]);
    // Optional
    connx_Tensor* axes = (input_count > 3) ? connx_Graph_get(graph, inputs[3]) : NULL;
    connx_Tensor* steps = (input_count > 4) ? connx_Graph_get(graph, inputs[4]) : NULL;

    int32_t output_ndim = data->ndim;
    int32_t output_shape[output_ndim];
    int32_t normalised_starts[data->ndim];
    int32_t normalised_ends[data->ndim];
    int32_t normalised_steps[data->ndim];

    // Calculate output shape
    memcpy(output_shape, data->shape, sizeof(int32_t) * output_ndim);
    // prefill starts, steps with 0, 1
    memset(normalised_starts, 0, sizeof(int32_t) * output_ndim);
    for (int32_t i = 0; i < output_ndim; i++) {
        normalised_ends[i] = output_shape[i];
        normalised_steps[i] = 1;
    }

    switch (starts->dtype) {
        /*{% for DTYPE, TYPE in loop_types(*supported_index_types) %}*/
    case {{ DTYPE }}: {
        {{TYPE}}* starts_array = starts->buffer;
        {{TYPE}}* ends_array = ends->buffer;

        {{TYPE}}* axes_array = (axes != NULL) ? axes->buffer : NULL;
        {{TYPE}}* steps_array = (steps != NULL) ? steps->buffer : NULL;

        // Reset slices
        const int64_t loop_count = (axes != NULL) ? axes->shape[0] : data->ndim;

        for (int64_t index = 0; index < loop_count; index++) {
            int32_t axis = (axes != NULL) ? axes_array[index] : index;
            int32_t start = starts_array[index];
            int32_t end = ends_array[index];
            int32_t step = (steps != NULL) ? steps_array[index] : 1;

            if (axis < 0) {
                axis += output_ndim;
            }

            if (start < 0) {
                start += data->shape[axis];
            }
            if (end < 0) {
                end += data->shape[axis];
            }

            start = clip(start, 0, data->shape[axis] - 1);
            end = clip(end, 0, data->shape[axis]);

            // if step is negative, then the slice is reversed
            int32_t axis_size;
            if (step > 0) {
                axis_size = ((end - start) / step) + ((end - start) % step > 0 ? 1 : 0);
            } else if (step < 0) {
                axis_size = ((start - end) / -step) + ((start - end) % -step > 0 ? 1 : 0);
            } else {
                // step is 0
                return CONNX_OUT_OF_INDEX;
            }

            normalised_starts[axis] = start;
            normalised_steps[axis] = step;

            output_shape[axis] = axis_size;
        }

        break;
    }
        /*{% endfor %}*/
    default:
        connx_error("Unsupported index type: %d", starts->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    // Create output tensor
    connx_Tensor* output = connx_Tensor_alloc(data->dtype, output_ndim, output_shape);
    if (output == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], output);

    // Get data type size
    size_t data_type_size;

    switch (data->dtype) {
        /*{% for dtype in supported_data_types %}*/
    case {{ dtype }}:
        /*{% endfor %}*/
        {
            data_type_size = connx_DataType_size(data->dtype);
            break;
        }
    default:
        connx_error("Slice: Datatype %d is not supported yet.\n", data->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    int32_t total = connx_Int32_product(output_ndim, output_shape);

    // Somethimes, one of axis is 0, which means total is 0
    // If without this check, the following loop will be endless
    if (total == 0) {
        return CONNX_OK;
    }

    int64_t batch_size = 1;
    for (int32_t i = output_ndim - 1; i >= 0; i--) {
        if (normalised_steps[i] == 1) {
            batch_size *= output_shape[i];
            if (normalised_starts[i] != 0 || normalised_ends[i] != output_shape[i]) {
                break;
            }
        } else {
            break;
        }
    }

    for (int64_t output_offset = 0; output_offset < total; output_offset += batch_size) {
        int64_t input_offset =
            get_input_index(output_ndim, data->shape, output_shape, normalised_starts, normalised_steps, output_offset);
        memcpy(output->buffer + output_offset * data_type_size, data->buffer + input_offset * data_type_size,
               batch_size * data_type_size);
    }

    return CONNX_OK;
}

int32_t get_input_index(const int32_t ndim, const int32_t* input_shape, const int32_t* output_shape,
                        const int32_t* starts, const int32_t* steps, const int32_t output_offset) {
    int32_t remaining = output_offset;
    int32_t input_offset = 0;

    int32_t input_indexes[ndim];
    int32_t output_indexes[ndim];

    for (int32_t i = 0; i < ndim; i++) {
        output_indexes[ndim - i - 1] = remaining % output_shape[ndim - i - 1];
        remaining /= output_shape[ndim - i - 1];
    }

    for (int32_t i = 0; i < ndim; i++) {
        input_indexes[i] = starts[i] + (output_indexes[i] * steps[i]);
    }

    for (int32_t i = 0; i < ndim; i++) {
        input_offset *= input_shape[i];
        input_offset += input_indexes[i];
    }

    return input_offset;
}
