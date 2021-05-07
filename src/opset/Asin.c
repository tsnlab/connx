#include <math.h>
#include "connx.h"

int Asin(connx_Graph* graph, uint32_t* outputs, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    connx_Tensor* input = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* output = connx_Tensor_alloc_like(input);

    int32_t total = connx_Int32_product(input->ndim, input->shape);

    switch(input->dtype) {
        case FLOAT32:
            {
                float32_t* input_array = input->buffer;
                float32_t* output_array = output->buffer;

                for(int32_t i = 0; i < total; i++) {
                    output_array[i] = asinf(input_array[i]);
                }
            }
            break;
        case FLOAT64:
            {
                float64_t* input_array = input->buffer;
                float64_t* output_array = output->buffer;

                for(int32_t i = 0; i < total; i++) {
                    output_array[i] = asin(input_array[i]);
                }
            }
            break;
        case FLOAT16:
        default:
            return NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], output);

    return OK;
}
