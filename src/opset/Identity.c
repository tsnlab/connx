#include <connx/accel.h>
#include <connx/connx.h>

int Identity(connx_Graph* graph, uint32_t output_count, uint32_t* outputs, uint32_t input_count, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    connx_Tensor* input = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* output = connx_Tensor_copy(input);

    if(output == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], output);

    return CONNX_OK;
}
