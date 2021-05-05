#include <stdio.h>
#include <string.h>
#include "tensor.h"

int main(__attribute__((unused)) int argc, char** argv) {
    connx_set_location(argv[1]);

    connx_Model model;
    connx_Model_init(&model);
    printf("model.version: %d\n", model.version);
    printf("model.opset_count: %u\n", model.opset_count);
    for(uint32_t i = 0; i < model.opset_count; i++) {
        printf("\tmodel.opset_names[%u] = '%s'\n", i, model.opset_names[i]);
        printf("\tmodel.opset_versions[%u] = '%u'\n", i, model.opset_versions[i]);
    }
    connx_Model_destroy(&model);

    return 0;
}
