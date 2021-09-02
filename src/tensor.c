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
#include <inttypes.h>
#include <math.h>   // sin, cos, ...
#include <stdlib.h> // strtol
#include <string.h> // memcpy

#include <connx/accel.h>
#include <connx/hal.h>
#include <connx/tensor.h>

uint32_t connx_DataType_size(connx_DataType dtype) {
    switch (dtype) {
    case CONNX_UINT8:
    case CONNX_INT8:
    case CONNX_BOOL:
        return 1;
    case CONNX_UINT16:
    case CONNX_INT16:
    case CONNX_FLOAT16:
        return 2;
    case CONNX_UINT32:
    case CONNX_INT32:
    case CONNX_FLOAT32:
        return 4;
    case CONNX_UINT64:
    case CONNX_INT64:
    case CONNX_FLOAT64:
    case CONNX_COMPLEX64:
        return 8;
    case CONNX_COMPLEX128:
        return 16;
    case CONNX_STRING:
        return sizeof(uintptr_t);
    case CONNX_UNDEFINED:
    default:
        return 0;
    }
}

int connx_Slice_init(connx_Slice* slice, int32_t start, int32_t end, int32_t step, int32_t idx) {
    slice->start = start;
    slice->end = end;
    slice->step = step;
    slice->idx = idx;

    return CONNX_OK;
}

// Iterator
void connx_Iterator_init(connx_Iterator* iterator) {
    int32_t ndim = iterator->ndim;
    connx_Slice* slices = iterator->slices;

    for (int32_t i = 0; i < ndim; i++) {
        slices[i].idx = slices[i].start;
    }

    slices[ndim - 1].idx -= slices[ndim - 1].step;
}

bool connx_Iterator_next(connx_Iterator* iterator) {
    int32_t ndim = iterator->ndim;
    connx_Slice* slices = iterator->slices;

    // Go next step
    for (int32_t i = ndim - 1; i >= 0; i--) {
        slices[i].idx += slices[i].step;
        if (slices[i].step > 0 && slices[i].idx < slices[i].end)
            return true;
        else if (slices[i].step < 0 && slices[i].idx > slices[i].end)
            return true;
        else
            slices[i].idx = slices[i].start;
    }

    // End of iterator
    connx_Iterator_init(iterator);

    return false;
}

int32_t connx_Iterator_offset(connx_Iterator* iterator, int32_t* shape) {
    int32_t ndim = iterator->ndim;
    connx_Slice* slices = iterator->slices;

    int32_t offset = 0;
    int32_t unit = 1;

    for (int32_t i = ndim - 1; i >= 0; i--) {
        offset += unit * slices[i].idx;
        unit *= shape[i];
    }

    return offset;
}

/**
 * Tensor payload: [connx_Tensor] [shape] [buffer]
 * All the elements are aligned by CONNX_ALIGNMENT
 */
connx_Tensor* connx_Tensor_alloc(connx_DataType dtype, int32_t ndim, int32_t* shape) {
    uint32_t header_size = CONNX_ALIGN(sizeof(connx_Tensor));
    uint32_t dim_size = CONNX_ALIGN(sizeof(int32_t) * ndim);
    int32_t total = connx_Int32_product(ndim, shape);
    uint32_t data_size = connx_DataType_size(dtype) * total;
    uint32_t buffer_size = CONNX_ALIGN(data_size);

    void* ptr = connx_alloc(header_size + dim_size + buffer_size);
    if (ptr == NULL) {
        return NULL;
    }

    connx_Tensor* tensor = ptr;
    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->shape = ptr + header_size;
    memcpy(tensor->shape, shape, sizeof(int32_t) * ndim);
    tensor->buffer = ptr + header_size + dim_size;
    tensor->size = data_size;
    tensor->parent = NULL;
    tensor->ref_count = 1;
    tensor->child_count = 0;
    connx_Lock_init(&tensor->lock);

    return tensor;
}

connx_Tensor* connx_Tensor_alloc_like(connx_Tensor* tensor) {
    return connx_Tensor_alloc(tensor->dtype, tensor->ndim, tensor->shape);
}

connx_Tensor* connx_Tensor_alloc_buffer(void* buf) {
    void* p = buf;

    uint32_t dtype = *(uint32_t*)p;
    p += sizeof(uint32_t);

    uint32_t ndim = *(uint32_t*)p;
    p += sizeof(uint32_t);

    int32_t shape[ndim];
    for (uint32_t i = 0; i < ndim; i++) {
        shape[i] = *(int32_t*)p;
        p += sizeof(int32_t);
    }

    uint32_t dsize = connx_DataType_size(dtype);
    uint32_t total = connx_Int32_product(ndim, shape);

    connx_Tensor* tensor = connx_Tensor_alloc(dtype, ndim, shape);
    if (tensor == NULL) {
        return NULL;
    }

    memcpy(tensor->buffer, p, dsize * total);

    return tensor;
}

#define next_token(token)                        \
    ({                                           \
        char* start = token;                     \
        while (*token != '_' && *token != '.') { \
            token++;                             \
        }                                        \
        *token++ = '\0';                         \
        start;                                   \
    })

#define next_integer(token)               \
    ({                                    \
        char* number = next_token(token); \
        if (number == NULL)               \
            return NULL;                  \
        strtol(number, NULL, 0);          \
    })

connx_Tensor* connx_Tensor_copy(connx_Tensor* tensor) {
    connx_Tensor* tensor2 = connx_Tensor_alloc_like(tensor);
    if (tensor2 == NULL)
        return NULL;

    int32_t total = connx_Int32_product(tensor->ndim, tensor->shape);
    int32_t data_size = connx_DataType_size(tensor->dtype);
    memcpy(tensor2->buffer, tensor->buffer, total * data_size);

    return tensor2;
}

connx_Tensor* connx_Tensor_reshape(connx_Tensor* tensor, int32_t ndim, int32_t* shape) {
    uint32_t header_size = CONNX_ALIGN(sizeof(connx_Tensor));
    uint32_t dim_size = CONNX_ALIGN(sizeof(int32_t) * ndim);

    void* ptr = connx_alloc(header_size + dim_size);
    if (ptr == NULL) {
        return NULL;
    }

    connx_Tensor* tensor2 = ptr;
    tensor2->dtype = tensor->dtype;
    tensor2->ndim = ndim;
    tensor2->shape = ptr + header_size;
    memcpy(tensor2->shape, shape, sizeof(int32_t) * ndim);
    tensor2->buffer = tensor->buffer;
    tensor2->size = tensor->size;
    tensor2->parent = tensor;
    tensor2->ref_count = 1;
    tensor2->child_count = 0;
    connx_Lock_init(&tensor2->lock);

    connx_Tensor_ref_child(tensor); // reshaped tensor references parent tensor

    return tensor2;
}

void connx_Tensor_ref(connx_Tensor* tensor) {
    connx_Lock_lock(&tensor->lock);

    tensor->ref_count++;

    connx_Lock_unlock(&tensor->lock);
}

int32_t connx_Tensor_unref(connx_Tensor* tensor) {
    int32_t ref_count = 0;

    connx_Lock_lock(&tensor->lock);

    ref_count = --tensor->ref_count;
    if (ref_count <= 0 && tensor->child_count <= 0) {
        connx_Tensor* parent = tensor->parent;

        connx_Lock_unlock(&tensor->lock);
        connx_Lock_destroy(&tensor->lock);

        connx_free(tensor);

        // Unref parent
        if (parent != NULL) {
            connx_Tensor_unref_child(parent);
        }

        return ref_count;
    }

    connx_Lock_unlock(&tensor->lock);

    return ref_count;
}

void connx_Tensor_ref_child(connx_Tensor* tensor) {
    connx_Lock_lock(&tensor->lock);

    tensor->child_count++;

    connx_Lock_unlock(&tensor->lock);
}

int32_t connx_Tensor_unref_child(connx_Tensor* tensor) {
    int32_t child_count = 0;

    connx_Lock_lock(&tensor->lock);

    child_count = --tensor->child_count;
    if (child_count <= 0 && tensor->ref_count <= 0) {
        connx_Tensor* parent = tensor->parent;

        connx_Lock_unlock(&tensor->lock);
        connx_Lock_destroy(&tensor->lock);

        connx_free(tensor);

        // Unref parent
        if (parent != NULL) {
            connx_Tensor_unref_child(parent);
        }

        return child_count;
    }

    connx_Lock_unlock(&tensor->lock);

    return child_count;
}

int connx_Tensor_get(connx_Tensor* tensor, connx_Iterator* iterator, void* data) {
    int32_t offset = connx_Iterator_offset(iterator, tensor->shape);
    uint32_t data_size = connx_DataType_size(tensor->dtype);
    memcpy(data, tensor->buffer + offset * data_size, data_size);

    return CONNX_OK;
}

int connx_Tensor_set(connx_Tensor* tensor, connx_Iterator* iterator, void* data) {
    int32_t offset = connx_Iterator_offset(iterator, tensor->shape);
    uint32_t data_size = connx_DataType_size(tensor->dtype);
    memcpy(tensor->buffer + offset * data_size, data, data_size);

    return CONNX_OK;
}

connx_Tensor* connx_Tensor_get_by_slice(connx_Tensor* tensor, connx_Slice* slices) {
    connx_Iterator tensor_iter = {tensor->ndim, slices};
    connx_Iterator_init(&tensor_iter);

    // Make a new tensor
    int32_t sliced_shape[tensor->ndim];
    for (int32_t i = 0; i < tensor->ndim; i++) {
        int32_t start = slices[i].start;
        int32_t end = slices[i].end;
        int32_t step = slices[i].step;
        int32_t diff = end - start;

        if (step == 0) {
            return NULL;
        }

        sliced_shape[i] = diff / step + (diff % step > 0 ? 1 : 0);
    }

    connx_Tensor* sliced = connx_Tensor_alloc(tensor->dtype, tensor->ndim, sliced_shape);

    if (sliced == NULL) {
        return NULL;
    }

    int32_t units[tensor->ndim];
    units[tensor->ndim - 1] = 1;
    for (int32_t i = tensor->ndim - 2; i >= 0; i--) {
        units[i] = units[i + 1] * tensor->shape[i + 1];
    }

    int32_t sliced_offset = 0;
    uint32_t data_size = connx_DataType_size(tensor->dtype);

    while (connx_Iterator_next(&tensor_iter)) {
        int32_t d_offset = 0;
        for (int i = 0; i < tensor->ndim; i++) {
            d_offset += units[i] * tensor_iter.slices[i].idx;
        }
        memcpy(sliced->buffer + sliced_offset * data_size, tensor->buffer + d_offset * data_size, data_size);
        sliced_offset++;
    }

    return sliced;
}

static int _connx_Tensor_set_by_slice(connx_Tensor* tensor, connx_Slice* slices, connx_Tensor* rhs) {
    for (int32_t i = 0; i < tensor->ndim; i++) {
        int32_t step = slices[i].step;

        if (step == 0) {
            return CONNX_OUT_OF_INDEX;
        }
    }

    connx_Iterator tensor_iter = {tensor->ndim, slices};
    connx_Iterator_init(&tensor_iter);

    int32_t units[tensor->ndim];
    units[tensor->ndim - 1] = 1;
    for (int32_t i = tensor->ndim - 2; i >= 0; i--) {
        units[i] = units[i + 1] * tensor->shape[i + 1];
    }

    int32_t rhs_offset = 0;
    while (connx_Iterator_next(&tensor_iter)) {
        int32_t tensor_offset = 0;

        for (int i = 0; i < tensor->ndim; i++) {
            tensor_offset += units[i] * tensor_iter.slices[i].idx;
        }

        uint32_t data_size = connx_DataType_size(tensor->dtype);
        memcpy(tensor->buffer + tensor_offset * data_size, rhs->buffer + rhs_offset * data_size, data_size);
        rhs_offset++;
    }

    return CONNX_OK;
}

static int _connx_Tensor_set_by_slice_tensor(connx_Tensor* tensor, connx_Slice* slices, connx_Tensor* rhs,
                                             connx_Slice* rhs_slices) {
    // Step size check
    for (int32_t i = 0; i < tensor->ndim; i++) {
        if (slices[i].step == 0) {
            return CONNX_OUT_OF_INDEX;
        }
    }

    for (int32_t i = 0; i < rhs->ndim; i++) {
        if (rhs_slices[i].step == 0) {
            return CONNX_OUT_OF_INDEX;
        }
    }

    connx_Iterator tensor_iter = {tensor->ndim, slices};
    connx_Iterator_init(&tensor_iter);

    connx_Iterator rhs_iter = {rhs->ndim, rhs_slices};
    connx_Iterator_init(&rhs_iter);

    int32_t tensor_units[tensor->ndim];
    tensor_units[tensor->ndim - 1] = 1;
    for (int32_t i = tensor->ndim - 2; i >= 0; i--) {
        tensor_units[i] = tensor_units[i + 1] * tensor->shape[i + 1];
    }

    int32_t rhs_units[rhs->ndim];
    rhs_units[rhs->ndim - 1] = 1;
    for (int32_t i = rhs->ndim - 2; i >= 0; i--) {
        rhs_units[i] = rhs_units[i + 1] * rhs->shape[i + 1];
    }

    while (connx_Iterator_next(&tensor_iter) && connx_Iterator_next(&rhs_iter)) {
        int32_t tensor_offset = 0;
        int32_t rhs_offset = 0;

        for (int i = 0; i < tensor->ndim; i++) {
            tensor_offset += tensor_units[i] * tensor_iter.slices[i].idx;
        }

        for (int i = 0; i < rhs->ndim; i++) {
            rhs_offset += rhs_units[i] * rhs_iter.slices[i].idx;
        }

        uint32_t data_size = connx_DataType_size(tensor->dtype);
        memcpy(tensor->buffer + tensor_offset * data_size, rhs->buffer + rhs_offset * data_size, data_size);
        rhs_offset++;
    }

    return CONNX_OK;
}

int connx_Tensor_set_by_slice(connx_Tensor* tensor, connx_Slice* slices, connx_Tensor* rhs, connx_Slice* rhs_slices) {
    if (rhs_slices == NULL) {
        return _connx_Tensor_set_by_slice(tensor, slices, rhs);
    } else {
        return _connx_Tensor_set_by_slice_tensor(tensor, slices, rhs, rhs_slices);
    }
}
