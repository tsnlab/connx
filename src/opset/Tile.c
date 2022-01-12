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
#include <string.h>

#ifdef DEBUG_TILE
#include <stdio.h>
#endif

#include <connx/accel.h>
#include <connx/connx.h>

static inline void get_indices(int32_t ndim, int32_t* shape, int32_t offset, int32_t* indices) {
    for (int32_t i = ndim - 1; i >= 0; i--) {
        indices[i] = offset % shape[i];
        offset /= shape[i];
    }
}

static void copy_input(void* output_buffer, void* input_buffer, int32_t ndim, int64_t* repeats_shape,
                       int32_t* input_shape, size_t type_size);

int Tile(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
         __attribute__((unused)) uint32_t input_count, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    /*{% set supported_dtypes = [
        UINT8, UINT16, UINT32, UINT64,
        INT8, INT16, INT32, INT64,
        FLOAT32, FLOAT64,
        BOOL,
    ]
    %}*/
    // TODO: STRING,

    // inputs
    connx_Tensor* input = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* repeats = connx_Graph_get(graph, inputs[1]);

    assert(repeats->ndim == 1 && repeats->shape[0] == input->ndim);

    int64_t* repeats_array = repeats->buffer;

    int32_t ndim_output = input->ndim;
    // Calculate output shape
    int32_t shape_output[ndim_output];

    for (int32_t i = 0; i < ndim_output; i++) {
        shape_output[i] = input->shape[i] * repeats_array[i];
    }

    // create output
    connx_Tensor* output = connx_Tensor_alloc(input->dtype, ndim_output, shape_output);

    if (output == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    size_t datatype_size;

    // get datatype size
    switch (input->dtype) {
        /*{% for DTYPE, TYPE in loop_types(*supported_dtypes) %}*/
    case {{ DTYPE }}:
        /*{% endfor %}*/
        datatype_size = connx_DataType_size(input->dtype);
        break;
    default:
        connx_error("Tile: Datatype %d is not supported yet.\n", input->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    assert(datatype_size > 0);
#ifdef DEBUG_TILE
    connx_Tensor_dump_header(input);
    connx_Tensor_dump_header(output);
    connx_Tensor_dump(repeats);
#endif

    // do tiling

    copy_input(output->buffer, input->buffer, input->ndim, repeats_array, input->shape, datatype_size);

    connx_Graph_set(graph, outputs[0], output);

    return CONNX_OK;
}

static inline void fill_rest(void* output_buffer, void* input_buffer, int32_t chunk_size, int32_t repeats,
                             size_t type_size) {
    for (int32_t i = 1; i < repeats; i++) {
        int32_t output_offset = i * chunk_size;
        memcpy(output_buffer + output_offset * type_size, input_buffer, chunk_size * type_size);
    }
}

#ifdef DEBUG_TILE
static inline void padding(int32_t ndim) {
    for (int i = 0; i < 5 - ndim; i++) {
        fprintf(stderr, "  ");
    }
}
#endif

void copy_input(void* output_buffer, void* input_buffer, int32_t ndim, int64_t* repeats_shape, int32_t* input_shape,
                size_t type_size) {
    int32_t repeats = repeats_shape[0];

    int32_t input_chunk_size = connx_Int32_product(ndim - 1, input_shape + 1);
    __attribute__((unused)) int32_t input_total_size = input_chunk_size * input_shape[0];
    int32_t output_chunk_size = connx_Int64_product(ndim - 1, repeats_shape + 1) * input_chunk_size;
    __attribute__((unused)) int32_t output_total_count = connx_Int64_product(ndim, repeats_shape);
    int32_t output_first_size = output_chunk_size * input_shape[0];

    if (ndim == 1) {
        // Copy whole array at once!
        input_chunk_size = input_shape[0];
        output_chunk_size = repeats * input_chunk_size;
        output_first_size = input_chunk_size;
    }

#ifdef DEBUG_TILE
    padding(ndim);
    fprintf(stderr, "ndim(%d): output_chunk: %d, input_chunk: %d, repeats: %d, output_total_count: %d\n", ndim,
            output_chunk_size, input_chunk_size, repeats, output_total_count);
#endif

    int32_t output_offset = 0;
    int32_t input_offset = 0;
    while (output_offset < output_first_size) {
#ifdef DEBUG_TILE
        padding(ndim);
        fprintf(stderr, "ndim(%d): Copying %d objects from %d to %d\n", ndim, input_chunk_size, input_offset,
                output_offset);
#endif
        if (ndim == 1) {
            memcpy(output_buffer + output_offset * type_size, input_buffer + input_offset * type_size,
                   input_chunk_size * type_size);
        } else {
            copy_input(output_buffer + output_offset * type_size, input_buffer + input_offset * type_size, ndim - 1,
                       repeats_shape + 1, input_shape + 1, type_size);
        }
        output_offset += output_chunk_size;
        input_offset += input_chunk_size;
    }
#ifdef DEBUG_TILE
    padding(ndim);
    fprintf(stderr, "ndim(%d): Fill rest %d times with chunk size %d\n", ndim, repeats - 1, output_chunk_size);
#endif
    fill_rest(output_buffer, output_buffer, output_first_size, repeats, type_size);
}
