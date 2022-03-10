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
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include <connx/accel.h>
#include <connx/connx.h>
#include <connx/hal_common.h>
#include <connx/opset.h>

#include "ver.h"

extern int connx_version_major;
extern int connx_version_minor;
extern int connx_version_micro;
extern char* connx_version_commit;
extern char* connx_version;

int connx_version_major = CONNX_VERSION_MAJOR;
int connx_version_minor = CONNX_VERSION_MINOR;
int connx_version_micro = CONNX_VERSION_MICRO;
char* connx_version_commit = CONNX_VERSION_COMMIT;
char* connx_version = CONNX_VERSION;

static char* _strdup(char* str) {
    int len = strlen(str);
    char* str2 = connx_alloc(len + 1);
    if (str2 == NULL)
        return NULL;

    memcpy(str2, str, len + 1);

    return str2;
}

#define next_token(token)                         \
    ({                                            \
        char* start = token;                      \
        while (*token != ' ' && *token != '\n') { \
            token++;                              \
        }                                         \
        *token++ = '\0';                          \
        start;                                    \
    })

#define check_keyword(token, keyword)                                 \
    ({                                                                \
        char* name = next_token(token);                               \
        if (strcmp(name, keyword) != 0) {                             \
            connx_error("Illegal syntax: %s != %s\n", name, keyword); \
            return CONNX_ILLEGAL_SYNTAX;                              \
        }                                                             \
    })

#define next_integer(token)                                            \
    ({                                                                 \
        char* number = next_token(token);                              \
        if (number == NULL) {                                          \
            connx_error("Illegal integer format before: %s\n", token); \
            return CONNX_ILLEGAL_SYNTAX;                               \
        }                                                              \
        strtol(number, NULL, 0);                                       \
    })

#define next_float(token)                                            \
    ({                                                               \
        char* number = next_token(token);                            \
        if (number == NULL) {                                        \
            connx_error("Illegal float format before: %s\n", token); \
            return CONNX_ILLEGAL_SYNTAX;                             \
        }                                                            \
        strtof(number, NULL);                                        \
    })

#define next_string(token)                  \
    ({                                      \
        uint32_t len = next_integer(token); \
        char* str = token;                  \
        str[len] = '\0';                    \
        token += len + 1;                   \
        str;                                \
    })

#define skip_comment(token)      \
    ({                           \
        while (*token != '\n') { \
            token++;             \
        }                        \
        token++;                 \
    })

static int parse_Model(connx_Model* model, char* metadata) {
    char* token = metadata;

    // prase version
    check_keyword(token, "connx");

    model->version = next_integer(token);

    if (model->version != 5) {
        connx_error("Not supported CONNX version: %u\n", model->version);
        return CONNX_NOT_SUPPORTED_CONNX_VERSION;
    }

    // parse opset_import
    check_keyword(token, "opset_import");
    model->opset_count = next_integer(token);

    model->opset_names = connx_alloc(sizeof(char*) * model->opset_count);
    if (model->opset_names == NULL) {
        connx_error("Out of memory\n");
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    model->opset_versions = connx_alloc(sizeof(uint32_t) * model->opset_count);
    if (model->opset_versions == NULL) {
        connx_error("Out of memory\n");
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    // parse opset name, version
    for (uint32_t i = 0; i < model->opset_count; i++) {
        model->opset_names[i] = _strdup(next_string(token));
        if (model->opset_names[i] == NULL) {
            connx_error("Out of memory\n");
            return CONNX_NOT_ENOUGH_MEMORY;
        }
        model->opset_versions[i] = next_integer(token);
    }

    // parse graph
    check_keyword(token, "graph");
    model->graph_count = next_integer(token);

    return CONNX_OK;
}

int connx_Model_init(connx_Model* model) {
    // Parse model
    void* metadata = connx_load_model();
    int ret = parse_Model(model, metadata);
    connx_unload_model(metadata);

    if (ret != CONNX_OK) {
        return ret;
    }

    // Parse graph
    model->graphs = connx_alloc(sizeof(connx_Graph*) * model->graph_count);
    if (model->graphs == NULL) {
        connx_error("Out of memory\n");
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    for (uint32_t i = 0; i < model->graph_count; i++) {
        model->graphs[i] = connx_alloc(sizeof(connx_Graph));
        if (model->graphs[i] == NULL) {
            connx_error("Out of memory\n");
            return CONNX_NOT_ENOUGH_MEMORY;
        }

        ret = connx_Graph_init(model->graphs[i], model, i);

        if (ret != CONNX_OK) {
            return ret;
        }
    }

    return CONNX_OK;
}

int connx_Model_destroy(connx_Model* model) {
    if (model->graphs != NULL) {
        for (uint32_t i = 0; i < model->graph_count; i++) {
            if (model->graphs[i] != NULL) {
                connx_Graph_destroy(model->graphs[i]);
                connx_free(model->graphs[i]);
            }
        }
        connx_free(model->graphs);
    }

    if (model->opset_versions != NULL) {
        connx_free(model->opset_versions);
    }

    if (model->opset_names != NULL) {
        for (uint32_t i = 0; i < model->opset_count; i++) {
            if (model->opset_names[i] != NULL) {
                connx_free(model->opset_names[i]);
            }
        }

        connx_free(model->opset_names);
    }

    return CONNX_OK;
}

int connx_Model_run(connx_Model* model, uint32_t input_count, connx_Tensor** inputs, uint32_t* output_count,
                    connx_Tensor** outputs) {
    return connx_Graph_run(model->graphs[0], input_count, inputs, output_count, outputs);
}

static int parse_initializer(connx_Tensor** tensor, uint32_t graph_id, uint32_t initializer_id) {
    void* buf = connx_load_data(graph_id, initializer_id);
    if (buf == NULL) {
        return CONNX_RESOURCE_NOT_FOUND;
    }

    *tensor = connx_Tensor_alloc_buffer(buf);
    if (*tensor == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_unload_data(buf);

    return CONNX_OK;
}

static int parse_Graph(connx_Graph* graph, char* text) {
    char* token = text;

    // prase value_info
    check_keyword(token, "value_info");

    graph->value_info_count = next_integer(token);
    graph->value_infos = connx_alloc(sizeof(connx_Tensor*) * (graph->value_info_count + 1)); // 0 is null
    if (graph->value_infos == NULL) {
        connx_error("Out of memory\n");
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    // prase initializer
    check_keyword(token, "initializer");

    graph->initializer_count = next_integer(token);
    graph->initializers = connx_alloc(sizeof(connx_Tensor*) * graph->initializer_count);
    if (graph->initializers == NULL) {
        connx_error("Out of memory\n");
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    for (uint32_t i = 0; i < graph->initializer_count; i++) {
        connx_Tensor* tensor;

        int ret = parse_initializer(&tensor, graph->id, i + 1);
        if (ret != CONNX_OK) {
            return ret;
        }

        graph->initializers[i] = tensor;
        connx_Tensor_ref_child(tensor);
        connx_Tensor_unref(tensor);
    }

    // prase output
    check_keyword(token, "output");

    graph->output_count = next_integer(token);
    graph->outputs = connx_alloc(sizeof(uint32_t) * graph->output_count);
    if (graph->outputs == NULL) {
        connx_error("Out of memory\n");
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    for (uint32_t i = 0; i < graph->output_count; i++) {
        graph->outputs[i] = next_integer(token);
    }

    // prase input
    check_keyword(token, "input");

    graph->input_count = next_integer(token);
    graph->inputs = connx_alloc(sizeof(uint32_t) * graph->input_count);
    if (graph->inputs == NULL) {
        connx_error("Out of memory\n");
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    for (uint32_t i = 0; i < graph->input_count; i++) {
        graph->inputs[i] = next_integer(token);
    }

    // prase node
    check_keyword(token, "node");

    graph->node_count = next_integer(token);
    graph->nodes = connx_alloc(sizeof(connx_Node*) * graph->node_count);
    if (graph->nodes == NULL) {
        connx_error("Out of memory\n");
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    for (uint32_t i = 0; i < graph->node_count; i++) {
        connx_Node* node = graph->nodes[i] = connx_alloc(sizeof(connx_Node));

        char* op_type = next_token(token);
        node->op_type = _strdup(op_type);
        if (node->op_type == NULL) {
            connx_error("Out of memory\n");
            return CONNX_NOT_ENOUGH_MEMORY;
        }

        // Find operator for op_type
        for (uint32_t i = 0; connx_opset_names[i] != NULL; i++) {
            if (strcmp(node->op_type, connx_opset_names[i]) == 0) {
                node->op = connx_opset_ops[i];
                break;
            }
        }

        if (node->op == NULL) {
            connx_error("Operator '%s' is not supported yet.\n", node->op_type);
            return CONNX_NOT_SUPPORTED_OPERATOR;
        }

        node->output_count = next_integer(token);
        node->input_count = next_integer(token);
        node->attribute_count = next_integer(token);

        node->outputs = connx_alloc(sizeof(uint32_t) * node->output_count);
        if (node->outputs == NULL) {
            connx_error("Out of memory\n");
            return CONNX_NOT_ENOUGH_MEMORY;
        }

        node->inputs = connx_alloc(sizeof(uint32_t) * node->input_count);
        if (node->inputs == NULL) {
            connx_error("Out of memory\n");
            return CONNX_NOT_ENOUGH_MEMORY;
        }

        node->attributes = connx_alloc(sizeof(uintptr_t) * node->attribute_count);
        if (node->attributes == NULL) {
            connx_error("Out of memory\n");
            return CONNX_NOT_ENOUGH_MEMORY;
        }

        for (uint32_t i = 0; i < node->output_count; i++) {
            node->outputs[i] = next_integer(token);
        }

        for (uint32_t i = 0; i < node->input_count; i++) {
            node->inputs[i] = next_integer(token);
        }

        // Parse attribute
        for (uint32_t i = 0; i < node->attribute_count; i++) {
            uint32_t type = next_integer(token);

            switch (type) {
            case 0: // NULL
                node->attributes[i] = NULL;
                break;
            case 1: // FLOAT
                node->attributes[i] = connx_alloc(sizeof(float32_t));
                if (node->attributes[i] == NULL) {
                    connx_error("Out of memory\n");
                    return CONNX_NOT_ENOUGH_MEMORY;
                }
                *(float32_t*)node->attributes[i] = next_float(token);
                break;
            case 2: // INT
                node->attributes[i] = connx_alloc(sizeof(int32_t));
                if (node->attributes[i] == NULL) {
                    connx_error("Out of memory\n");
                    return CONNX_NOT_ENOUGH_MEMORY;
                }
                *(int32_t*)node->attributes[i] = next_integer(token);
                break;
            case 3: // STRING
                node->attributes[i] = _strdup(next_string(token));
                if (node->attributes[i] == NULL) {
                    connx_error("Out of memory\n");
                    return CONNX_NOT_ENOUGH_MEMORY;
                }
                break;
            case 6: { // FLOATS
                uint32_t count = next_integer(token);
                connx_AttributeFloats* attr =
                    connx_alloc(sizeof(connx_AttributeFloats) + sizeof(float32_t) * count); // count, array
                if (attr == NULL) {
                    connx_error("Out of memory\n");
                    return CONNX_NOT_ENOUGH_MEMORY;
                }

                node->attributes[i] = attr;
                attr->count = count;
                for (uint32_t j = 0; j < count; j++) {
                    attr->array[j] = next_float(token);
                }
                break;
            }
            case 7: { // INTS
                uint32_t count = next_integer(token);
                connx_AttributeInts* attr =
                    connx_alloc(sizeof(connx_AttributeInts) + sizeof(int32_t) * count); // count, array

                if (attr == NULL) {
                    connx_error("Out of memory\n");
                    return CONNX_NOT_ENOUGH_MEMORY;
                }

                node->attributes[i] = attr;
                attr->count = count;
                for (uint32_t j = 0; j < count; j++) {
                    attr->array[j] = next_integer(token);
                }
                break;
            }
            case 8: { // STRINGS
                uint32_t count = next_integer(token);

                uint32_t size = 0;
                char* strs[count];

                for (uint32_t j = 0; j < count; j++) {
                    strs[j] = next_string(token);
                    size += strlen(strs[j]) + 1;
                }

                connx_AttributeStrings* attr = connx_alloc(sizeof(connx_AttributeStrings) + sizeof(char*) * count +
                                                           size); // count, array, string buffer

                if (attr == NULL) {
                    connx_error("Out of memory\n");
                    return CONNX_NOT_ENOUGH_MEMORY;
                }

                node->attributes[i] = attr;
                attr->count = count;
                char* buf = (void*)attr->array[count]; // Point the next to the array
                for (uint32_t j = 0; j < count; j++) {
                    attr->array[j] = buf;

                    size_t len = strlen(strs[j]) + 1;
                    memcpy(buf, strs[j], len);
                    buf += len;
                }
                break;
            }
            default:
                connx_error("Attribute type %u is not supported.\n", type);
                return CONNX_NOT_SUPPORTED_ATTRIBUTE;
            }
        }

        if (*token == '#') {
#ifdef DEBUG
            next_token(token); // skip '#'

            node->name = _strdup(next_string(token));

            node->output_names = connx_alloc(sizeof(char*) * node->output_count);
            for (uint32_t i = 0; i < node->output_count; i++) {
                node->output_names[i] = _strdup(next_string(token));
            }

            node->input_names = connx_alloc(sizeof(char*) * node->input_count);
            for (uint32_t i = 0; i < node->input_count; i++) {
                node->input_names[i] = _strdup(next_string(token));
            }

            node->attribute_names = connx_alloc(sizeof(char*) * node->attribute_count);
            for (uint32_t i = 0; i < node->attribute_count; i++) {
                node->attribute_names[i] = _strdup(next_string(token));
            }
#else
            skip_comment(token);
#endif /* DEBUG */
        }
    }

    return CONNX_OK;
}

int connx_Graph_init(connx_Graph* graph, connx_Model* model, uint32_t graph_id) {
    graph->model = model;
    graph->id = graph_id;

    // Parse value_info
    void* text = connx_load_text(graph_id);
    int ret = parse_Graph(graph, text);
    connx_unload_text(text);

    if (ret != CONNX_OK) {
        return ret;
    }

    return CONNX_OK;
}

int connx_Graph_destroy(connx_Graph* graph) {
    if (graph->nodes != NULL) {
        for (uint32_t i = 0; i < graph->node_count; i++) {
            if (graph->nodes[i] != NULL) {
                connx_Node* node = graph->nodes[i];
                if (node->op_type != NULL) {
                    connx_free(node->op_type);
                }

                if (node->attributes != NULL) {
                    for (uint32_t j = 0; j < node->attribute_count; j++) {
                        if (node->attributes[j] != NULL) {
                            connx_free(node->attributes[j]);
                        }
                    }
                    connx_free(node->attributes);
                }

                if (node->inputs != NULL) {
                    connx_free(node->inputs);
                }

                if (node->outputs != NULL) {
                    connx_free(node->outputs);
                }

#ifdef DEBUG
                if (node->attribute_names != NULL) {
                    for (uint32_t i = 0; i < node->attribute_count; i++) {
                        connx_free(node->attribute_names[i]);
                    }
                    connx_free(node->attribute_names);
                }

                if (node->input_names != NULL) {
                    for (uint32_t i = 0; i < node->input_count; i++) {
                        connx_free(node->input_names[i]);
                    }
                    connx_free(node->input_names);
                }

                if (node->output_names != NULL) {
                    for (uint32_t i = 0; i < node->output_count; i++) {
                        connx_free(node->output_names[i]);
                    }
                    connx_free(node->output_names);
                }

                if (node->name != NULL) {
                    connx_free(node->name);
                }
#endif /* DEBUG */

                connx_free(node);
            }
        }
        connx_free(graph->nodes);
    }

    if (graph->value_infos != NULL) {
        for (uint32_t i = 0; i < graph->value_info_count; i++) {
            if (graph->value_infos[i] != NULL) {
                connx_Tensor_unref(graph->value_infos[i]);
            }
        }
        connx_free(graph->value_infos);
    }

    if (graph->outputs != NULL) {
        connx_free(graph->outputs);
    }

    if (graph->inputs != NULL) {
        connx_free(graph->inputs);
    }

    if (graph->initializers != NULL) {
        for (uint32_t i = 0; i < graph->initializer_count; i++) {
            if (graph->initializers[i] != NULL) {
                connx_Tensor_unref_child(graph->initializers[i]);
            }
        }
        connx_free(graph->initializers);
    }

    return CONNX_OK;
}

int connx_Graph_run(connx_Graph* graph, uint32_t input_count, connx_Tensor** inputs, uint32_t* output_count,
                    connx_Tensor** outputs) {
    // Set inputs
    input_count = input_count < graph->input_count ? input_count : graph->input_count;

    for (uint32_t i = 0; i < input_count; i++) {
        uint32_t id = graph->inputs[i];
        graph->value_infos[id] = inputs[i];
    }

    // Initialize value_infos
    for (uint32_t i = 0; i < graph->initializer_count; i++) {
        graph->value_infos[i + 1] = graph->initializers[i];
        connx_Tensor_ref(graph->value_infos[i + 1]);
    }

    // Execute operators
    for (uint32_t i = 0; i < graph->node_count; i++) {
        connx_Node* node = graph->nodes[i];
        int ret = node->op(graph, node->output_count, node->outputs, node->input_count, node->inputs,
                           node->attribute_count, node->attributes);
        if (ret != CONNX_OK) {
            return ret;
        }

#if DEBUG
        connx_dump_node_outputs(graph, node);
#endif /* DEBUG */

        for (uint32_t i = 0; i < node->input_count; i++) {
            uint32_t id = node->inputs[i];
            connx_Tensor* tensor = graph->value_infos[id];
            // TODO: It's a temporary fix
            if (tensor != NULL) {
                int32_t ref_count = connx_Tensor_unref(tensor);
                if (ref_count <= 0) {
                    graph->value_infos[id] = NULL;
                }
            }
        }
    }

    // Set outputs
    *output_count = *output_count < graph->output_count ? *output_count : graph->output_count;
    for (uint32_t i = 0; i < *output_count; i++) {
        uint32_t id = graph->outputs[i];
        outputs[i] = graph->value_infos[id];
        graph->value_infos[id] = NULL;
    }

    // Clean value_infos
    for (uint32_t i = 0; i < graph->value_info_count; i++) {
        if (graph->value_infos[i] != NULL) {
            connx_Tensor_unref(graph->value_infos[i]);
        }
    }

    bzero(graph->value_infos, graph->value_info_count * sizeof(uint32_t));

    return CONNX_OK;
}

connx_Tensor* connx_Graph_get(connx_Graph* graph, uint32_t id) {
    return graph->value_infos[id];
}

void connx_Graph_set(connx_Graph* graph, uint32_t id, connx_Tensor* tensor) {
    if (graph->value_infos[id] == tensor)
        return;

    if (graph->value_infos[id] != NULL) {
        connx_Tensor_unref(graph->value_infos[id]);
    }

    graph->value_infos[id] = tensor;
}
