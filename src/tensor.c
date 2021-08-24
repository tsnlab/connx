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
#include <math.h> // sin, cos, ...
#include <stdio.h>
#include <stdlib.h> // strtol
#include <string.h> // memcpy

#include <connx/accel.h>
#include <connx/hal.h>
#include <connx/tensor.h>

TEMPLATE_START(FLOAT32, FLOAT64, UINT32, UINT64, INT32, INT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t

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
        if ((slices[i].step > 0 && slices[i].idx < slices[i].stop) || slices[i].idx > slices[i].stop)
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
    connx_Lock_init(&tensor2->lock);

    connx_Tensor_ref(tensor); // reshaped tensor references parent tensor

    return tensor2;
}

void connx_Tensor_ref(connx_Tensor* tensor) {
    connx_Lock_lock(&tensor->lock);

    tensor->ref_count++;

    connx_Lock_unlock(&tensor->lock);
}

void connx_Tensor_unref(connx_Tensor* tensor) {
    connx_Lock_lock(&tensor->lock);

    if (--tensor->ref_count <= 0) {
        connx_Tensor* parent = tensor->parent;

        connx_Lock_unlock(&tensor->lock);
        connx_Lock_destroy(&tensor->lock);

        connx_free(tensor);

        // Unref parent
        if (parent != NULL) {
            connx_Tensor_unref(parent);
        }

        return;
    }

    connx_Lock_unlock(&tensor->lock);
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

int connx_Slice_set(connx_Slice* slice, int32_t start, int32_t stop, int32_t step) {
    slice->start = start;
    slice->stop = stop;
    slice->step = step;

    return CONNX_OK;
}

connx_Tensor* connx_Tensor_get_by_slice(connx_Tensor* tensor, connx_Slice* slices) {
    connx_Iterator tensor_iter = {tensor->ndim, slices};
    connx_Iterator_init(&tensor_iter);

    // Make new tensor
    int32_t sliced_shape[tensor->ndim];
    for (int32_t i = 0; i < tensor->ndim; i++) {
        // start 또는 stop이 음수인 경우 shape을 더해서 인덱스를 양수로 바꾼다.
        int32_t start = slices[i].start >= 0 ? slices[i].start : tensor->shape[i] + slices[i].start;
        int32_t stop = slices[i].stop >= 0 ? slices[i].stop : tensor->shape[i] + slices[i].stop;
        int32_t step = slices[i].step > 0 ? slices[i].step : -slices[i].step;
        step = slices[i].step == 0 ? 1 : step;
        fprintf(stderr, "슬라이스 쉐입[%d] %d:%d:%d = %d => ", i, start, stop, step, stop - start);

        // start보다 stop이 큰 경우 둘다 0 처리하여 범위 지정하지 않는다.
        if (start > stop) {
            start = 0;
            stop = 0;
        }
        // 슬라이싱 시작지점이 쉐입을 넘는 경우 최소 크기를 지정한다.
        else if (start > tensor->shape[i]) {
            start = tensor->shape[i];
        }

        // 슬라이싱 하는 stop 구간이 쉐입 범위를 넘어서는 경우 최대 크기를 지정한다.
        else if (stop > tensor->shape[i]) {
            stop = tensor->shape[i];
        }

        fprintf(stderr, "슬라이스 쉐입[%d] %d:%d:%d = %d \n", i, start, stop, step, stop - start);
        sliced_shape[i] = ceilf((float)(stop - start) / (float)step);
    }
    connx_Tensor* sliced_tensor = connx_Tensor_alloc(tensor->dtype, tensor->ndim, sliced_shape);

    if (sliced_tensor == NULL) {
        return NULL;
    }

    int32_t tensor_units[tensor->ndim];
    tensor_units[tensor->ndim - 1] = 1;
    for (int32_t i = tensor->ndim - 2; i >= 0; i--) {
        tensor_units[i] = tensor_units[i + 1] * tensor->shape[i + 1];
    }

    TEMPLATE_TYPE* sliced_tensor_array = sliced_tensor->buffer;
    TEMPLATE_TYPE* tensor_array = tensor->buffer;

    int32_t sliced_offset = 0;
    while (connx_Iterator_next(&tensor_iter)) {
        int32_t d_idx[tensor->ndim];
        for (int i = 0; i < tensor->ndim; i++) {
            d_idx[i] = tensor_iter.slices[i].idx;
        }

        int32_t d_offset = 0;
        for (int32_t i = 0; i < tensor->ndim; i++) {
            d_offset += tensor_units[i] * d_idx[i];
        }

        sliced_tensor_array[sliced_offset++] = tensor_array[d_offset];
    }

    return sliced_tensor;
}

int connx_Tensor_set_by_slice(connx_Tensor* lhs, connx_Slice* slices, connx_Tensor* rhs) {
    // 대입 받는 측, 대입하는 측 모두 이터레이션 필요
    // 두 이터레이터는 레인지나 스텝은 달라도 쉐입 같음
    connx_Iterator lhs_iter = {lhs->ndim, slices};
    connx_Iterator_init(&lhs_iter);

    int32_t lhs_units[lhs->ndim];

    lhs_units[lhs->ndim - 1] = 1;
    for (int32_t i = lhs->ndim - 2; i >= 0; i--) {
        lhs_units[i] = lhs_units[i + 1] * lhs->shape[i + 1];
    }

    TEMPLATE_TYPE* lhs_array = lhs->buffer;
    TEMPLATE_TYPE* rhs_array = rhs->buffer;

    int32_t rhs_offset = 0;
    while (connx_Iterator_next(&lhs_iter)) {
        int32_t lhs_d_idx[lhs->ndim];

        // 이터레이터를 이용한 인덱스 계산
        for (int i = 0; i < lhs->ndim; i++) {
            lhs_d_idx[i] = lhs_iter.slices[i].idx;
        }

        // 인데스를 가지고 1차원 상에서 offset 계산
        int32_t lhs_offset = 0;
        for (int32_t i = 0; i < lhs->ndim; i++) {
            lhs_offset += lhs_units[i] * lhs_d_idx[i];
        }
        lhs_array[lhs_offset] = rhs_array[rhs_offset++];
    }

    return CONNX_OK;
}
