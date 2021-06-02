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
    OK = 0,
    DATA_TYPE_NOT_MATCHING = 1,
    TENSOR_SHAPE_NOT_MATCHING = 2,
    ILLEGAL_SYNTAX = 3,
    NOT_SUPPORTED_CONNX_VERSION = 4,
    NOT_SUPPORTED_OPERATOR = 5,
    NOT_SUPPORTED_ATTRIBUTE = 6,
    NOT_SUPPORTED_DATATYPE = 7,
    NOT_ENOUGH_MEMORY = 8,
    RESOURCE_NOT_FOUND = 9,
    IO_ERROR = 10,
} connx_ErrorCode;

// The number of the enumeration follow ONNX's TensorProto.DataType code
typedef enum _connx_DataType {
    UNDEFINED = 0,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    UINT32 = 12,
    INT32 = 6,
    UINT64 = 13,
    INT64 = 7,
    FLOAT16 = 10,
    FLOAT32 = 1,
    FLOAT64 = 11,
    STRING = 8,
    BOOL = 9,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
} connx_DataType;

uint32_t connx_DataType_size(connx_DataType dtype);

// Template engine
#define TEMPLATE_START(...)
#define TEMPLATE_END(...)
#define TEMPLATE_DTYPE
#define TEMPLATE_TYPE
#define TEMPLATE_NAME

#endif /* __CONNX_TYPES_H__ */
