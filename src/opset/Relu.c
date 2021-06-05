#include <connx/accel.h>
#include <connx/connx.h>

int Relu(connx_Graph* graph, uint32_t output_count, uint32_t* outputs, uint32_t input_count, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* Y = connx_Tensor_alloc_like(X);
    if(Y == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t total = connx_Int32_product(X->ndim, X->shape);

    switch(X->dtype) {
        TEMPLATE_START(FLOAT32, INT32, INT8, INT16, INT64, FLOAT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE FLOAT32
#define TEMPLATE_TYPE float32_t
        case TEMPLATE_DTYPE: {
            TEMPLATE_TYPE* X_array = X->buffer;
            TEMPLATE_TYPE* Y_array = Y->buffer;

            for(int32_t i = 0; i < total; i++) {
                Y_array[i] = X_array[i] < 0 ? 0 : X_array[i];
            }
            break;
        }
            TEMPLATE_END()
        default:
            connx_error("Relu: Datatype %d is not supported yet.\n", X->dtype);
            return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return CONNX_OK;
}
