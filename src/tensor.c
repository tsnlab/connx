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
#include <inttypes.h>
#include <math.h>   // sin, cos, ...
#include <stdlib.h> // strtol
#include <string.h> // memcpy

#include <connx/accel.h>
#include <connx/hal_common.h>
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

bool connx_Tensor_is_likely_same_shape(connx_Tensor* A, connx_Tensor* B) {
    int32_t ndim_big = A->ndim > B->ndim ? A->ndim : B->ndim;
    int32_t ndim_small = A->ndim < B->ndim ? A->ndim : B->ndim;

    int32_t pad_a = ndim_big - A->ndim;
    int32_t pad_b = ndim_big - B->ndim;

    // Prepended dimensions must be 1, 1, 1
    for (int32_t i = 0; i < pad_a; i += 1) {
        if (A->shape[i] != 1) {
            return false;
        }
    }
    for (int32_t i = 0; i < pad_b; i += 1) {
        if (B->shape[i] != 1) {
            return false;
        }
    }

    // All the rest dimensions must be equal
    for (int32_t i = 0; i < ndim_small; i += 1) {
        if (A->shape[i + pad_a] != B->shape[i + pad_b]) {
            return false;
        }
    }

    return true;
}

connx_Tensor* connx_Tensor_alloc_broadcasted(const connx_DataType dtype, connx_Tensor* A, connx_Tensor* B) {
    int32_t ndim = (A->ndim > B->ndim) ? A->ndim : B->ndim;
    int32_t shape[ndim];

    int32_t padding_a = ndim - A->ndim;
    int32_t padding_b = ndim - B->ndim;

    for (int32_t i = 0; i < ndim; i++) {
        if (padding_a > i) {
            shape[i] = B->shape[i];
        } else if (padding_b > i) {
            shape[i] = A->shape[i];
        } else {
            int32_t len_a = A->shape[i - padding_a];
            int32_t len_b = B->shape[i - padding_b];
            if (len_a == len_b) {
                shape[i] = len_a;
            } else if (len_a == 1) {
                shape[i] = len_b;
            } else if (len_b == 1) {
                shape[i] = len_a;
            } else {
                connx_error("Tensor broadcast failed: shape mismatch\n");
                return NULL;
            }
        }
    }

    return connx_Tensor_alloc(dtype, ndim, shape);
}

int32_t connx_Tensor_get_broadcasted_input_offset(const connx_Tensor* output, const connx_Tensor* input,
                                                  int32_t output_offset) {
    int32_t* output_shape = output->shape;
    int32_t* input_shape = input->shape;
    int32_t output_idxs[output->ndim];
    int32_t input_offset = 0;

    int32_t skip_size = output->ndim - input->ndim;

    // Skip first dimensions if input has smaller dimensions
    output_shape += skip_size;
    int32_t ndim = output->ndim - skip_size;

    // Calculate output indices
    for (int32_t i = ndim - 1; i >= 0; i--) {
        output_idxs[i] = output_offset % output_shape[i];
        output_offset /= output_shape[i];
    }

    // Calculate input indices
    for (int32_t i = 0; i < input->ndim; i++) {
        input_offset *= input_shape[i];
        assert(output_shape[i] >= input_shape[i]);

        if (input_shape[i] == 1) {
            input_offset += 0;
        } else {
            input_offset += output_idxs[i];
        }
    }

    return input_offset;
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
