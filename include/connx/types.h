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
#ifndef __CONNX_TYPES_H__
#define __CONNX_TYPES_H__

#include <float.h>
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
    CONNX_OUT_OF_INDEX = 3,
    CONNX_ILLEGAL_SYNTAX = 4,
    CONNX_NOT_SUPPORTED_CONNX_VERSION = 5,
    CONNX_NOT_SUPPORTED_OPERATOR = 6,
    CONNX_NOT_SUPPORTED_ATTRIBUTE = 7,
    CONNX_NOT_SUPPORTED_DATATYPE = 8,
    CONNX_NOT_SUPPORTED_FEATURE = 9,
    CONNX_NOT_ENOUGH_MEMORY = 10,
    CONNX_RESOURCE_NOT_FOUND = 11,
    CONNX_IO_ERROR = 12,
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

uint32_t connx_DataType_size(uint32_t dtype /** @see connx_DataType */);

#define UINT8 CONNX_UINT8
#define INT8 CONNX_INT8
#define UINT16 CONNX_UINT16
#define INT16 CONNX_INT16
#define UINT32 CONNX_UINT32
#define INT32 CONNX_INT32
#define UINT64 CONNX_UINT64
#define INT64 CONNX_INT64
#define FLOAT16 CONNX_FLOAT16
#define FLOAT32 CONNX_FLOAT32
#define FLOAT64 CONNX_FLOAT64
#define STRING CONNX_STRING
#define BOOL CONNX_BOOL
#define COMPLEX64 CONNX_COMPLEX64
#define COMPLEX128 CONNX_COMPLEX128

#define CONNX_INT8_MIN INT8_MIN
#define CONNX_INT8_MAX INT8_MAX
#define CONNX_INT16_MIN INT16_MIN
#define CONNX_INT16_MAX INT16_MAX
#define CONNX_INT32_MIN INT32_MIN
#define CONNX_INT32_MAX INT32_MAX
#define CONNX_INT64_MIN INT64_MIN
#define CONNX_INT64_MAX INT64_MAX
#define CONNX_UINT8_MIN 0
#define CONNX_UINT8_MAX UINT8_MAX
#define CONNX_UINT16_MIN 0
#define CONNX_UINT16_MAX UINT16_MAX
#define CONNX_UINT32_MIN 0
#define CONNX_UINT32_MAX UINT32_MAX
#define CONNX_UINT64_MIN 0
#define CONNX_UINT64_MAX UINT64_MAX
#define CONNX_FLOAT16_MIN -65504
#define CONNX_FLOAT16_MAX 65504
#define CONNX_FLOAT32_MIN -FLT_MAX
#define CONNX_FLOAT32_MAX FLT_MAX
#define CONNX_FLOAT64_MIN -DBL_MAX
#define CONNX_FLOAT64_MAX DBL_MAX

#endif /* __CONNX_TYPES_H__ */
