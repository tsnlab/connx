#include <math.h>
#include "connx.h"

int Asin(connx_Graph* graph, uint32_t* outputs, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    connx_Tensor* input = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* output = connx_Tensor_alloc_like(input);

    int32_t total = connx_Int32_product(input->ndim, input->shape);

    switch(input->dtype) {
TEMPLATE_START(FLOAT32, float32_t, FLOAT64, float64_t)
#undef _DTYPE
#undef _TYPE
#define _DTYPE FLOAT32
#define _TYPE float32_t
        case _DTYPE:
            {
                _TYPE* input_array = input->buffer;
                _TYPE* output_array = output->buffer;

                for(int32_t i = 0; i < total; i++) {
                    output_array[i] = asinf(input_array[i]);
                }
            }
            break;
TEMPLATE_END()
        default:
            connx_error("Asin: Datatype %d is not supported yet.\n", input->dtype);
            return NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], output);

    return OK;
}
