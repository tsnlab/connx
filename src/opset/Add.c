#include <connx/accel.h>
#include <connx/connx.h>

int Add(connx_Graph* graph, __attribute__((unused)) uint32_t output_count, uint32_t* outputs, __attribute__((unused)) uint32_t input_count, uint32_t* inputs, __attribute__((unused)) void** attributes) {
    connx_Tensor* A = connx_Graph_get(graph, inputs[0]);
    connx_Tensor* B = connx_Graph_get(graph, inputs[1]);

    int32_t ndim = A->ndim > B->ndim ? A->ndim : B->ndim;
    int32_t shape[ndim];
    for(int32_t i = 0; i < ndim; i++) {
        int32_t A_dim = i < A->ndim ? A->shape[ndim - i - 1] : 0;
        int32_t B_dim = i < B->ndim ? B->shape[ndim - i - 1] : 0;
        shape[ndim - i - 1] = A_dim > B_dim ? A_dim : B_dim;
    }

    connx_Tensor* C = connx_Tensor_alloc(A->dtype, ndim, shape);

    if(C == NULL) {
        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t A_total = connx_Int32_product(A->ndim, A->shape);
    int32_t B_total = connx_Int32_product(B->ndim, B->shape);
    int32_t C_total = connx_Int32_product(C->ndim, C->shape);

    switch(A->dtype) {
        TEMPLATE_START(UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FLOAT32, FLOAT64)
#undef TEMPLATE_DTYPE
#undef TEMPLATE_TYPE
#define TEMPLATE_DTYPE INT32
#define TEMPLATE_TYPE int32_t
        case TEMPLATE_DTYPE: {
            TEMPLATE_TYPE* A_array = A->buffer;
            TEMPLATE_TYPE* B_array = B->buffer;
            TEMPLATE_TYPE* C_array = C->buffer;

            for(int32_t C_idx = 0, A_idx = 0, B_idx = 0; C_idx < C_total;
                C_idx++, A_idx = (A_idx + 1) % A_total, B_idx = (B_idx + 1) % B_total) {

                C_array[C_idx] = A_array[A_idx] + B_array[B_idx];
            }
            break;
        }
            TEMPLATE_END()
        default:
            connx_error("Add: Datatype %d is not supported yet.\n", A->dtype);
            return CONNX_NOT_SUPPORTED_DATATYPE;
    }

    connx_Graph_set(graph, outputs[0], C);

    return CONNX_OK;
}
