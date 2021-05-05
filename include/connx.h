#ifndef __CONNX_H__
#define __CONNX_H__

#include <stdint.h>

typedef enum _connx_ErrorCode {
    OK                          = 0,
    DATA_TYPE_NOT_MATCHING      = 1,
    TENSOR_SHAPE_NOT_MATCHING   = 2,
    ILLEGAL_SYNTAX              = 3,
    NOT_SUPPORTED_CONNX_VERSION = 4,
    NOT_SUPPORTED_OPERATOR      = 5,
    NOT_ENOUGH_MEMORY           = 6,
} connx_ErrorCode;

typedef struct _connx_Model {
    int32_t     version;
    uint32_t    opset_count;
    char**      opset_names;
    uint32_t*   opset_versions;
    uint32_t    graph_count;
} connx_Model;

int connx_Model_init(connx_Model* model);
int connx_Model_destroy(connx_Model* model);

#endif /* __CONNX_H__ */
