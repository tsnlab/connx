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
#ifndef __CONNX_TENSOR_H__
#define __CONNX_TENSOR_H__

#include <connx/hal.h>
#include <connx/types.h>

// tensor structure follow Numpy's ndarray
typedef struct _connx_Tensor {
    connx_DataType dtype;         // data type
    int32_t ndim;                 // Number of dimensions
    int32_t* shape;               // Shape array
    void* buffer;                 // Data buffer
    uint32_t size;                // size of buffer
    struct _connx_Tensor* parent; // Parent tensor that share the buffer
    int32_t ref_count;            // Reference count
    int32_t child_count;          // Child count
    connx_Lock lock;              // Reference and child count lock
} connx_Tensor;

connx_Tensor* connx_Tensor_alloc(connx_DataType dtype, int32_t ndim, int32_t* shape);
connx_Tensor* connx_Tensor_alloc_like(connx_Tensor* tensor);
connx_Tensor* connx_Tensor_alloc_buffer(void* buf);

/**
 * @brief Check if two tensor have same shape or one have prepended 1s
 * Example: [4, 7, 2] and [4, 7, 2] are same shape while [1, 2, 3] and [4, 7, 2] are not
 * [4, 7, 2] and [1, 4, 7, 2] are also same shape because [1] is prepended and same in array
 * But [4, 7, 2] and [2, 4, 7, 2] are not same shape
 *
 * @param A Input tensor A
 * @param B Input tensor B
 * @return true if two tensor have same shape or one have prepended 1s, or false
 */
bool connx_Tensor_is_likely_same_shape(connx_Tensor* A, connx_Tensor* B);

/**
 * Create broadcasted tensor
 *
 * @param dtype output tensor's data type
 * @param A a tensor
 * @param B another tensor
 * @return newly allocated tensor. NULL if cannot be broadcasted.
 */
connx_Tensor* connx_Tensor_alloc_broadcasted(const connx_DataType dtype, connx_Tensor* A, connx_Tensor* B);

/**
 * @brief Get the broadcasted input offset object
 * @see connx_Tensor_alloc_broadcasted
 *
 * @param output Output tensor that has broadcasted dimension
 * @param input Input tensor to calculate the offset
 * @param output_offset Offset to use on output tensor
 * @return int32_t offset to use for input tensor
 */
int32_t connx_Tensor_get_broadcasted_input_offset(const connx_Tensor* output, const connx_Tensor* input,
                                                  int32_t output_offset);

connx_Tensor* connx_Tensor_copy(connx_Tensor* tensor);
connx_Tensor* connx_Tensor_reshape(connx_Tensor* tensor, int32_t ndim, int32_t* shape);

void connx_Tensor_ref(connx_Tensor* tensor);
int32_t connx_Tensor_unref(connx_Tensor* tensor);
void connx_Tensor_ref_child(connx_Tensor* tensor);
int32_t connx_Tensor_unref_child(connx_Tensor* tensor);

#endif /* __CONNX_TENSOR_H__ */
