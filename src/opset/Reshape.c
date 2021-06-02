#include <connx/accel.h>
#include <connx/connx.h>

int Reshape(connx_Graph* graph, uint32_t* outputs, uint32_t* inputs, void** attributes) {
    connx_Tensor* data = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* shape = connx_Graph_get(graph, inputs[1]);
    int32_t* allowzero = attributes[0];

    int32_t ndim = shape->shape[0];
    int32_t new_shape[ndim];

    // Copy tensor shape to array new_shape
    int32_t negative_idx = -1;
    for(int32_t i = 0; i < ndim; i++) {
        new_shape[i] = ((int64_t*)shape->buffer)[i];

        if(*allowzero == 0 && new_shape[i] == 0) {
            new_shape[i] = data->shape[i];
        }

        if(new_shape[i] == -1) {
            negative_idx = i;
        }
    }

    // Process -1 dim
    if(negative_idx >= 0) {
        int32_t total = connx_Int32_product(data->ndim, data->shape);

        new_shape[negative_idx] = 1;
        int32_t remain = connx_Int32_product(ndim, new_shape);

        new_shape[negative_idx] = total / remain;
    }

    // Make a reshaped tensor
    connx_Tensor* reshaped = connx_Tensor_reshape(data, ndim, new_shape);
    if(reshaped == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    connx_Graph_set(graph, outputs[0], reshaped);

    return CONNX_OK;
}
