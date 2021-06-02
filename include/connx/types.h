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
