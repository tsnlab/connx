#ifndef __TYPES_H__
#define __TYPES_H__

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef uint16_t float16_t;
typedef float float32_t;
typedef double float64_t;

typedef enum _connx_ErrorCode {
    OK                          = 0,
    DATA_TYPE_NOT_MATCHING      = 1,
    TENSOR_SHAPE_NOT_MATCHING   = 2,
    ILLEGAL_SYNTAX              = 3,
    NOT_SUPPORTED_CONNX_VERSION = 4,
    NOT_SUPPORTED_OPERATOR      = 5,
    NOT_SUPPORTED_ATTRIBUTE     = 6,
    NOT_ENOUGH_MEMORY           = 7,
} connx_ErrorCode;

#endif /* __TYPES_H__ */
