#include <stdio.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "connx.h"
#include "opset.h"
#include "hal.h"

static char* _strdup(char* str) {
    int len = strlen(str);
    char* str2 = connx_alloc(len + 1);
    if(str2 == NULL)
        return NULL;

    memcpy(str2, str, len + 1);

    return str2;
}

#define next_token(token)                       \
    ({                                          \
        char* start = token;                    \
        while(*token != ' ' && *token != '\n') {\
            token++;                            \
        }                                       \
        *token++ = '\0';                        \
        start;                                  \
    })

#define check_keyword(token, keyword)       \
    ({                                      \
        char* name = next_token(token);     \
        if(strcmp(name, keyword) != 0) {    \
            connx_error("Illegal syntax: %s != %s\n", name, keyword); \
            return ILLEGAL_SYNTAX;          \
        }                                   \
    })

#define next_integer(token)                     \
    ({                                          \
        char* number = next_token(token);       \
        if(number == NULL) {                    \
            connx_error("Illegal integer format before: %s\n", token); \
            return ILLEGAL_SYNTAX;              \
        }                                       \
        strtol(number, NULL, 0);                \
    })

#define next_float(token)                       \
    ({                                          \
        char* number = next_token(token);       \
        if(number == NULL) {                    \
            connx_error("Illegal float format before: %s\n", token); \
            return ILLEGAL_SYNTAX;              \
        }                                       \
        strtof(number, NULL);                   \
    })

#define next_string(token)                      \
    ({                                          \
        uint32_t len = next_integer(token);     \
        char* str = token;                      \
        str[len] = '\0';                        \
        token += + len + 1;                     \
        str;                                    \
    })

static connx_ErrorCode parse_Model(connx_Model* model, char* metadata) {
    char* token = metadata;

    // prase version
    check_keyword(token, "connx");

    model->version = next_integer(token);

    if(model->version <= 0 || model->version > 1) {
        connx_error("Not supported CONNX version: %u\n", model->version);
        return NOT_SUPPORTED_CONNX_VERSION;
    }

    // parse opset_import
    check_keyword(token, "opset_import");
    model->opset_count = next_integer(token);

    model->opset_names = connx_alloc(sizeof(char*) * model->opset_count);
    if(model->opset_names == NULL) {
        connx_error("Out of memory\n");
        return NOT_ENOUGH_MEMORY;
    }

    model->opset_versions = connx_alloc(sizeof(uint32_t) * model->opset_count);
    if(model->opset_versions == NULL) {
        connx_error("Out of memory\n");
        return NOT_ENOUGH_MEMORY;
    }

    // parse opset name, version
    for(uint32_t i = 0; i < model->opset_count; i++) {
        model->opset_names[i] = _strdup(next_string(token));
        if(model->opset_names[i] == NULL) {
            connx_error("Out of memory\n");
            return NOT_ENOUGH_MEMORY;
        }
        model->opset_versions[i] = next_integer(token);
    }

    // parse graph
    check_keyword(token, "graph");
    model->graph_count = next_integer(token);

    return OK;
}

int connx_Model_init(connx_Model* model) {
    // Parse model
    void* metadata = connx_load("model.connx");
    connx_ErrorCode ret = parse_Model(model, metadata);
    connx_unload(metadata);

    if(ret != OK) {
        return ret;
    }

    // Parse graph
    model->graphs = connx_alloc(sizeof(connx_Graph*) * model->graph_count);
    if(model->graphs == NULL) {
        connx_error("Out of memory\n");
        return NOT_ENOUGH_MEMORY;
    }

    for(uint32_t i = 0; i < model->graph_count; i++) {
        model->graphs[i] = connx_alloc(sizeof(connx_Graph));
        if(model->graphs[i] == NULL) {
            connx_error("Out of memory\n");
            return NOT_ENOUGH_MEMORY;
        }

        ret = connx_Graph_init(model->graphs[i], model, i);

        if(ret != OK) {
            return ret;
        }
    }

    return OK;
}

int connx_Model_destroy(connx_Model* model) {
    if(model->graphs != NULL) {
        for(uint32_t i = 0; i < model->graph_count; i++) {
            if(model->graphs[i] != NULL) {
                connx_Graph_destroy(model->graphs[i]);
                connx_free(model->graphs[i]);
            }
        }
        connx_free(model->graphs);
    }

    if(model->opset_versions != NULL) {
        connx_free(model->opset_versions);
    }

    if(model->opset_names != NULL) {
        for(uint32_t i = 0; i < model->opset_count; i++) {
            if(model->opset_names[i] != NULL) {
                connx_free(model->opset_names[i]);
            }
        }

        connx_free(model->opset_names);
    }

    return OK;
}

int connx_Model_run(connx_Model* model, uint32_t input_count, connx_Tensor** inputs, uint32_t* output_count, connx_Tensor** outputs) {
    return connx_Graph_run(model->graphs[0], input_count, inputs, output_count, outputs);
}

static connx_ErrorCode parse_Graph(connx_Graph* graph, char* text) {
    char* token = text;

    // prase value_info
    check_keyword(token, "value_info");

    graph->value_info_count = next_integer(token);
    graph->value_infos = connx_alloc(sizeof(connx_Tensor*) * (graph->value_info_count + 1)); // 0 is null
    if(graph->value_infos == NULL) {
        connx_error("Out of memory\n");
        return NOT_ENOUGH_MEMORY;
    }

    // prase initializer
    check_keyword(token, "initializer");

    graph->initializer_count = next_integer(token);
    graph->initializers = connx_alloc(sizeof(connx_Tensor*) * graph->initializer_count);
    if(graph->initializers == NULL) {
        connx_error("Out of memory\n");
        return NOT_ENOUGH_MEMORY;
    }

    // prase output
    check_keyword(token, "output");

    graph->output_count = next_integer(token);
    graph->outputs = connx_alloc(sizeof(uint32_t) * graph->output_count);
    if(graph->outputs == NULL) {
        connx_error("Out of memory\n");
        return NOT_ENOUGH_MEMORY;
    }

    for(uint32_t i = 0; i < graph->output_count; i++) {
        graph->outputs[i] = next_integer(token);
    }

    // prase input
    check_keyword(token, "input");

    graph->input_count = next_integer(token);
    graph->inputs = connx_alloc(sizeof(uint32_t) * graph->input_count);
    if(graph->inputs == NULL) {
        connx_error("Out of memory\n");
        return NOT_ENOUGH_MEMORY;
    }

    for(uint32_t i = 0; i < graph->input_count; i++) {
        graph->inputs[i] = next_integer(token);
    }

    // prase node
    check_keyword(token, "node");

    graph->node_count = next_integer(token);
    graph->nodes = connx_alloc(sizeof(connx_Node*) * graph->node_count);
    if(graph->nodes == NULL) {
        connx_error("Out of memory\n");
        return NOT_ENOUGH_MEMORY;
    }

    for(uint32_t i = 0; i < graph->node_count; i++) {
        connx_Node* node = graph->nodes[i] = connx_alloc(sizeof(connx_Node));

        char* op_type = next_token(token);
        node->op_type = _strdup(op_type);
        if(node->op_type == NULL) {
            connx_error("Out of memory\n");
            return NOT_ENOUGH_MEMORY;
        }

        // Find operator for op_type
        for(uint32_t i = 0; connx_opset_names[i] != NULL; i++) {
            if(strcmp(node->op_type, connx_opset_names[i]) == 0) {
                node->op = connx_opset_ops[i];
                break;
            }
        }

        if(node->op == NULL) {
            connx_error("Operator %s is not supported yet.\n", node->op_type);
            return NOT_SUPPORTED_OPERATOR;
        }

        node->output_count = next_integer(token);
        node->input_count = next_integer(token);
        node->attribute_count = next_integer(token);

        node->outputs = connx_alloc(sizeof(uint32_t) * node->output_count);
        if(node->outputs == NULL) {
            connx_error("Out of memory\n");
            return NOT_ENOUGH_MEMORY;
        }

        node->inputs = connx_alloc(sizeof(uint32_t) * node->input_count);
        if(node->inputs == NULL) {
            connx_error("Out of memory\n");
            return NOT_ENOUGH_MEMORY;
        }

        node->attributes = connx_alloc(sizeof(uintptr_t) * node->attribute_count);
        if(node->attributes == NULL) {
            connx_error("Out of memory\n");
            return NOT_ENOUGH_MEMORY;
        }

        for(uint32_t i = 0; i < node->output_count; i++) {
            node->outputs[i] = next_integer(token);
        }

        for(uint32_t i = 0; i < node->input_count; i++) {
            node->inputs[i] = next_integer(token);
        }

        // Parse attribute
        for(uint32_t i = 0; i < node->attribute_count; i++) {
            next_string(token); // Drop name
            uint32_t type = next_integer(token);

            uint32_t count;
            switch(type) {
                case 1: // FLOAT
                    node->attributes[i] = connx_alloc(sizeof(float32_t));
                    if(node->attributes[i] == NULL) {
                        connx_error("Out of memory\n");
                        return NOT_ENOUGH_MEMORY;
                    }
                    *(float32_t*)node->attributes[i] = next_float(token);
                    break;
                case 2: // INT
                    node->attributes[i] = connx_alloc(sizeof(int32_t));
                    if(node->attributes[i] == NULL) {
                        connx_error("Out of memory\n");
                        return NOT_ENOUGH_MEMORY;
                    }
                    *(int32_t*)node->attributes[i] = next_integer(token);
                    break;
                case 3: // STRING
                    node->attributes[i] = _strdup(next_string(token));
                    if(node->attributes[i] == NULL) {
                        connx_error("Out of memory\n");
                        return NOT_ENOUGH_MEMORY;
                    }
                    break;
                case 6: // FLOATS
                    count = next_integer(token);
                    node->attributes[i] = connx_alloc(sizeof(float32_t) * count);
                    if(node->attributes[i] == NULL) {
                        connx_error("Out of memory\n");
                        return NOT_ENOUGH_MEMORY;
                    }

                    for(uint32_t j = 0; j < count; j++) {
                        ((float32_t*)node->attributes[i])[j] = next_float(token);
                    }
                    break;
                case 7: // INTS
                    count = next_integer(token);
                    node->attributes[i] = connx_alloc(sizeof(int32_t) * count);
                    if(node->attributes[i] == NULL) {
                        connx_error("Out of memory\n");
                        return NOT_ENOUGH_MEMORY;
                    }

                    for(uint32_t j = 0; j < count; j++) {
                        ((int32_t*)node->attributes[i])[j] = next_integer(token);
                    }
                    break;
                case 8: // STRINGS
                    count = next_integer(token);
                    node->attributes[i] = connx_alloc(sizeof(char*) * count);
                    if(node->attributes[i] == NULL) {
                        connx_error("Out of memory\n");
                        return NOT_ENOUGH_MEMORY;
                    }

                    for(uint32_t j = 0; j < count; j++) {
                        ((char**)node->attributes[i])[j] = _strdup(next_string(token));
                    }
                    break;
                default:
                    connx_error("Attribute type %u is not supported.\n", type);
                    return NOT_SUPPORTED_ATTRIBUTE;
            }
        }
    }

    return OK;
}

int connx_Graph_init(connx_Graph* graph, connx_Model* model, uint32_t graph_id) {
    graph->model = model;
    graph->id = graph_id;

    // Parse value_info
    char name[256];
    snprintf(name, 256, "%u.text", graph_id);

    void* text = connx_load(name);
    connx_ErrorCode ret = parse_Graph(graph, text);
    connx_unload(text);

    if(ret != OK) {
        return ret;
    }

    return OK;
}

int connx_Graph_destroy(connx_Graph* graph) {
    if(graph->nodes != NULL) {
        for(uint32_t i = 0; i < graph->node_count; i++) {
            if(graph->nodes[i] != NULL) {
                connx_Node* node = graph->nodes[i];
                if(node->op_type != NULL) {
                    connx_free(node->op_type);
                }

                if(node->attributes != NULL) {
                    for(uint32_t j = 0; j < node->attribute_count; j++) {
                        if(node->attributes[j] != NULL) {
                            connx_free(node->attributes[j]);
                        }
                    }
                    connx_free(node->attributes);
                }

                if(node->inputs != NULL) {
                    connx_free(node->inputs);
                }

                if(node->outputs != NULL) {
                    connx_free(node->outputs);
                }

                connx_free(node);
            }
        }
        connx_free(graph->nodes);
    }

    if(graph->value_infos != NULL) {
        for(uint32_t i = 0; i < graph->value_info_count; i++) {
            if(graph->value_infos[i] != NULL) {
                connx_Tensor_unref(graph->value_infos[i]);
            }
        }
        connx_free(graph->value_infos);
    }

    if(graph->outputs != NULL) {
        connx_free(graph->outputs);
    }

    if(graph->inputs != NULL) {
        connx_free(graph->inputs);
    }

    if(graph->initializers != NULL) {
        for(uint32_t i = 0; i < graph->initializer_count; i++) {
            if(graph->initializers[i] != NULL) {
                connx_Tensor_unref(graph->initializers[i]);
            }
        }
        connx_free(graph->initializers);
    }

    return OK;
}

int connx_Graph_run(connx_Graph* graph, uint32_t input_count, connx_Tensor** inputs, uint32_t* output_count, connx_Tensor** outputs) {
    // Set inputs
    input_count = input_count < graph->input_count ? input_count : graph->input_count;

    for(uint32_t i = 0; i < input_count; i++) {
        uint32_t id = graph->inputs[i];
        graph->value_infos[id] = inputs[i];
    }

    // Initialize value_infos
    for(uint32_t i = 0; i < graph->initializer_count; i++) {
        if(graph->value_infos[i + 1] != NULL) {
            graph->value_infos[i + 1] = connx_Tensor_copy(graph->initializers[i]);
        }
    }

    // Execute operators
    for(uint32_t i = 0; i < graph->node_count; i++) {
        connx_Node* node = graph->nodes[i];
        int ret = node->op(graph, node->outputs, node->inputs, node->attributes);
        if(ret != OK) {
            return ret;
        }
    }

    // Set outputs
    *output_count = *output_count < graph->output_count ? *output_count : graph->output_count;
    for(uint32_t i = 0; i < *output_count; i++) {
        uint32_t id = graph->outputs[i];
        outputs[i] = graph->value_infos[id];
        graph->value_infos[id] = NULL;
    }

    // Clean value_infos
    for(uint32_t i = 0; i < graph->value_info_count; i++) {
        if(graph->value_infos[i] != NULL) {
            connx_Tensor_unref(graph->value_infos[i]);
            graph->value_infos[i] = NULL;
        }
    }

    return OK;
}

connx_Tensor* connx_Graph_get(connx_Graph* graph, uint32_t id) {
    return graph->value_infos[id];
}

void connx_Graph_set(connx_Graph* graph, uint32_t id, connx_Tensor* tensor) {
    if(graph->value_infos[id] == tensor)
        return;

    if(graph->value_infos[id] != NULL) {
        connx_Tensor_unref(graph->value_infos[id]);
    }

    graph->value_infos[id] = tensor;
}

