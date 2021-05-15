#include "connx.h"

int Relu(connx_Graph* graph, uint32_t* outputs, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    connx_Tensor* X = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* Y = connx_Tensor_alloc_like(X);
    if(Y == NULL) {
        return NOT_ENOUGH_MEMORY;
    }

    int32_t total = connx_Int32_product(X->ndim, X->shape);

    switch(X->dtype) {
TEMPLATE_START(FLOAT32, float32_t, INT32, int32_t, INT8, int8_t, INT16, int16_t, INT64, int64_t, FLOAT64, float64_t)
#undef _DTYPE
#undef _TYPE
#define _DTYPE FLOAT32
#define _TYPE float32_t
        case _DTYPE:
            {
                _TYPE* X_array = X->buffer;
                _TYPE* Y_array = Y->buffer;

                for(int32_t i = 0; i < total; i++) {
                    Y_array[i] = X_array[i] < 0 ? 0 : X_array[i];
                }
            }
            break;
TEMPLATE_END()
        default:
            connx_error("Relu: Datatype %d is not supported yet.\n", X->dtype);
            return NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], Y);

    return OK;
}
