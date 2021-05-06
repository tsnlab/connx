#include <stdio.h>
#include <string.h>
#include "connx.h"

int main(__attribute__((unused)) int argc, char** argv) {
    connx_set_location(argv[1]);

    connx_Model model;
    int ret = connx_Model_init(&model);
    if(ret != 0) {
        printf("Error: %d\n", ret);
        return ret;
    }

    printf("model.version: %d\n", model.version);
    printf("model.opset_count: %u\n", model.opset_count);
    for(uint32_t i = 0; i < model.opset_count; i++) {
        printf("\tmodel.opset_names[%u] = '%s'\n", i, model.opset_names[i]);
        printf("\tmodel.opset_versions[%u] = '%u'\n", i, model.opset_versions[i]);
    }

    connx_Graph* graph = model.graphs[0];
    printf("graph.id: %u\n", graph->id);
    printf("graph.initializer_count: %u\n", graph->initializer_count);

    printf("graph.inputs: ");
    for(uint32_t i = 0; i < graph->input_count; i++) {
        printf("%u ", graph->inputs[i]);
    }
    printf("\n");

    printf("graph.outputs: ");
    for(uint32_t i = 0; i < graph->output_count; i++) {
        printf("%u ", graph->outputs[i]);
    }
    printf("\n");

    printf("graph.value_info_count: %u\n", graph->value_info_count);

    printf("graph.node_count: %u\n", graph->node_count);
    for(uint32_t i = 0; i < graph->node_count; i++) {
        connx_Node* node = graph->nodes[i];

        printf("\tnode[%u].op_type = '%s'\n", i, node->op_type);

        printf("\tnode[%u].outputs =", i);
        for(uint32_t j = 0; j < node->output_count; j++) {
            printf(" %u", node->outputs[j]);
        }
        printf("\n");

        printf("\tnode[%u].inputs =", i);
        for(uint32_t j = 0; j < node->input_count; j++) {
            printf(" %u", node->inputs[j]);
        }
        printf("\n");

        printf("node[%u].attribute_count = %u\n", i, node->attribute_count);
    }

    connx_Model_destroy(&model);

    return 0;
}
