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
#ifndef __CONNX_TENSOR_H__
#define __CONNX_TENSOR_H__

#include <connx/hal.h>
#include <connx/types.h>

// Iterator
typedef struct _connx_Slice {
    int32_t start; // Start index (>=)
    int32_t stop;  // Stop index (<)
    int32_t step;  // Step size
    int32_t idx;   // Current index (used by iterator)
} connx_Slice;

typedef struct _connx_Iterator {
    int32_t ndim;        // number of slices
    connx_Slice* slices; // slices
} connx_Iterator;

void connx_Iterator_init(connx_Iterator* iterator);
bool connx_Iterator_next(connx_Iterator* iterator);
/**
 * Get offset of data from linear array.
 *
 * @param iterator iterator for a tensor
 * @param shape shape of a tensor
 * @return offset of data which iterator is pointing
 */
int32_t connx_Iterator_offset(connx_Iterator* iterator, int32_t* shape);

// tensor structure follow Numpy's ndarray
typedef struct _connx_Tensor {
    connx_DataType dtype;         // data type
    int32_t ndim;                 // Number of dimensions
    int32_t* shape;               // Shape array
    void* buffer;                 // Data buffer
    uint32_t size;                // size of buffer
    struct _connx_Tensor* parent; // Parent tensor that share the buffer
    int32_t ref_count;            // Reference count
    connx_Lock lock;              // Reference lock
} connx_Tensor;

connx_Tensor* connx_Tensor_alloc(connx_DataType dtype, int32_t ndim, int32_t* shape);
connx_Tensor* connx_Tensor_alloc_like(connx_Tensor* tensor);
connx_Tensor* connx_Tensor_alloc_buffer(void* buf);
connx_Tensor* connx_Tensor_copy(connx_Tensor* tensor);
connx_Tensor* connx_Tensor_reshape(connx_Tensor* tensor, int32_t ndim, int32_t* shape);

void connx_Tensor_ref(connx_Tensor* tensor);
void connx_Tensor_unref(connx_Tensor* tensor);

/**
 * Get an element(data) which iterator is pointing.
 *
 * @param tensor getting a data from a tensor
 * @param iterator getting a data which iterator is pointing
 * @param data a pointer to copy an element from tensor
 * @return connx_ErrorCode
 */
int connx_Tensor_get(connx_Tensor* tensor, connx_Iterator* iterator, void* data);

/**
 * Set an element(data) which iterator is pointing.
 *
 * @param tensor setting a data from a tensor
 * @param iterator setting a data which iterator is pointing
 * @param data a pointer to copy an element to tensor
 * @return connx_ErrorCode
 */
int connx_Tensor_set(connx_Tensor* tensor, connx_Iterator* iterator, void* data);


int connx_Slice_set(connx_Slice* slice, int32_t start, int32_t stop, int32_t end);

connx_Tensor* connx_Tensor_get_by_slice(connx_Tensor* tensor, connx_Slice* slices);
int connx_Tensor_set_by_slice(connx_Tensor* lhs, connx_Slice* slices, connx_Tensor* rhs);
#endif /* __CONNX_TENSOR_H__ */
