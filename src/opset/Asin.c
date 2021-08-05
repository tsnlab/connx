#include <math.h>

#include <connx/accel.h>
#include <connx/connx.h>

int Asin(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs,
         __attribute__((unused)) uint32_t input_count, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    connx_Tensor* input = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* output = connx_Tensor_alloc_like(input);

    int32_t total = connx_Int32_product(input->ndim, input->shape);

    switch (input->dtype) {
        TEMPLATE_START(FLOAT32, FLOAT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
    case TEMPLATE_DTYPE: {
        TEMPLATE_TYPE* input_array = input->buffer;
        TEMPLATE_TYPE* output_array = output->buffer;

        for (int32_t i = 0; i < total; i++) {
            output_array[i] = asinf(input_array[i]);
        }
        break;
    }
        TEMPLATE_END()
    default:
        connx_error("Asin: Datatype %d is not supported yet.\n", input->dtype);
        return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], output);

    return CONNX_OK;
}
