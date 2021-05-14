#include "connx.h"

int Add(connx_Graph* graph, uint32_t* outputs, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    connx_Tensor* A = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* B = connx_Graph_get(graph, inputs[1]);

    int32_t ndim = A->ndim > B->ndim ? A->ndim : B->ndim;
    int32_t shape[ndim];
    for(int32_t i = 0; i < ndim; i++) {
        int32_t A_dim = i < A->ndim  ? A->shape[ndim - i - 1] : 0;
        int32_t B_dim = i < B->ndim  ? B->shape[ndim - i - 1] : 0;
        shape[ndim - i - 1] = A_dim > B_dim ? A_dim : B_dim;
    }

    connx_Tensor* C = connx_Tensor_alloc(A->dtype, ndim, shape);

    if(C == NULL) {
        return NOT_ENOUGH_MEMORY;
    }

    int32_t A_total = connx_Int32_product(A->ndim, A->shape);
    int32_t B_total = connx_Int32_product(B->ndim, B->shape);
    int32_t C_total = connx_Int32_product(C->ndim, C->shape);

    switch(A->dtype) {
        case UINT8:
            {
                uint8_t* A_array = A->buffer;
                uint8_t* B_array = B->buffer;
                uint8_t* C_array = C->buffer;

                for(int32_t C_idx = 0, A_idx = 0, B_idx = 0; 
                        C_idx < C_total;
                        C_idx++,
                        A_idx = (A_idx + 1) % A_total,
                        B_idx = (B_idx + 1) % B_total) {

                    C_array[C_idx] = A_array[A_idx] + B_array[B_idx];
                }
            }
            break;
        case FLOAT32:
            {
                float32_t* A_array = A->buffer;
                float32_t* B_array = B->buffer;
                float32_t* C_array = C->buffer;

                for(int32_t C_idx = 0, A_idx = 0, B_idx = 0; 
                        C_idx < C_total;
                        C_idx++,
                        A_idx = (A_idx + 1) % A_total,
                        B_idx = (B_idx + 1) % B_total) {

                    C_array[C_idx] = A_array[A_idx] + B_array[B_idx];
                }
            }
            break;
        case FLOAT64:
            {
                float64_t* A_array = A->buffer;
                float64_t* B_array = B->buffer;
                float64_t* C_array = C->buffer;

                for(int32_t C_idx = 0, A_idx = 0, B_idx = 0; 
                        C_idx < C_total;
                        C_idx++,
                        A_idx = (A_idx + 1) % A_total,
                        B_idx = (B_idx + 1) % B_total) {

                    C_array[C_idx] = A_array[A_idx] + B_array[B_idx];
                }
            }
            break;
        case FLOAT16:
        default:
            return NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], C);

    return OK;
}
