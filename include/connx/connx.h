#ifndef __CONNX_CONNX_H__
#define __CONNX_CONNX_H__

#include <connx/tensor.h>

typedef struct _connx_Graph connx_Graph;

typedef struct _connx_Model {
    int32_t version;

    uint32_t opset_count;
    char** opset_names;
    uint32_t* opset_versions;

    uint32_t graph_count;
    connx_Graph** graphs;
} connx_Model;

typedef int (*CONNX_OPERATOR)(connx_Graph* graph, uint32_t output_count, uint32_t* outputs, uint32_t input_count,
                              uint32_t* inputs, void** attributes);

typedef struct _connx_Node {
    uint32_t output_count;
    uint32_t* outputs;

    uint32_t input_count;
    uint32_t* inputs;

    uint32_t attribute_count;
    void** attributes;

    char* op_type;
    CONNX_OPERATOR op;
} connx_Node;

typedef struct _connx_AttributeFloats {
    uint32_t count;
    float32_t array[0];
} connx_AttributeFloats;

typedef struct _connx_AttributeInts {
    uint32_t count;
    int32_t array[0];
} connx_AttributeInts;

typedef struct _connx_AttributeStrings {
    uint32_t count;
    char* array[0];
} connx_AttributeStrings;

struct _connx_Graph {
    connx_Model* model;

    uint32_t id;

    uint32_t initializer_count;
    connx_Tensor** initializers;
    uint32_t input_count;
    uint32_t* inputs;

    uint32_t output_count;
    uint32_t* outputs;

    uint32_t value_info_count;
    connx_Tensor** value_infos;

    uint32_t node_count;
    connx_Node** nodes;
};

int connx_Model_init(connx_Model* model);
int connx_Model_destroy(connx_Model* model);
int connx_Model_run(connx_Model* model, uint32_t input_count, connx_Tensor** inputs, uint32_t* output_count,
                    connx_Tensor** outputs);

int connx_Graph_init(connx_Graph* graph, connx_Model* model, uint32_t graph_id);
int connx_Graph_destroy(connx_Graph* graph);
int connx_Graph_run(connx_Graph* graph, uint32_t input_count, connx_Tensor** inputs, uint32_t* output_count,
                    connx_Tensor** outputs);

connx_Tensor* connx_Graph_get(connx_Graph* graph, uint32_t id);
void connx_Graph_set(connx_Graph* graph, uint32_t id, connx_Tensor* tensor);

#endif /* __CONNX_CONNX_H__ */
