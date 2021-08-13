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
#ifndef __CONNX_TYPES_H__
#define __CONNX_TYPES_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef float float16_t;
typedef float float32_t;
typedef double float64_t;

typedef struct {
    float real;
    float imag;
} complex64_t;

typedef struct {
    double real;
    double imag;
} complex128_t;

typedef enum _connx_ErrorCode {
    CONNX_OK = 0,
    CONNX_DATA_TYPE_NOT_MATCHING = 1,
    CONNX_TENSOR_SHAPE_NOT_MATCHING = 2,
    CONNX_ILLEGAL_SYNTAX = 3,
    CONNX_NOT_SUPPORTED_CONNX_VERSION = 4,
    CONNX_NOT_SUPPORTED_OPERATOR = 5,
    CONNX_NOT_SUPPORTED_ATTRIBUTE = 6,
    CONNX_NOT_SUPPORTED_DATATYPE = 7,
    CONNX_NOT_ENOUGH_MEMORY = 8,
    CONNX_RESOURCE_NOT_FOUND = 9,
    CONNX_IO_ERROR = 10,
} connx_ErrorCode;

// The number of the enumeration follow ONNX's TensorProto.DataType code
typedef enum _connx_DataType {
    CONNX_UNDEFINED = 0,
    CONNX_UINT8 = 2,
    CONNX_INT8 = 3,
    CONNX_UINT16 = 4,
    CONNX_INT16 = 5,
    CONNX_UINT32 = 12,
    CONNX_INT32 = 6,
    CONNX_UINT64 = 13,
    CONNX_INT64 = 7,
    CONNX_FLOAT16 = 10,
    CONNX_FLOAT32 = 1,
    CONNX_FLOAT64 = 11,
    CONNX_STRING = 8,
    CONNX_BOOL = 9,
    CONNX_COMPLEX64 = 14,
    CONNX_COMPLEX128 = 15,
} connx_DataType;

uint32_t connx_DataType_size(connx_DataType dtype);

// Template engine
#define TEMPLATE_START(...)
#define TEMPLATE_END(...)
#define TEMPLATE_DTYPE
#define TEMPLATE_TYPE
#define TEMPLATE_NAME

#endif /* __CONNX_TYPES_H__ */
