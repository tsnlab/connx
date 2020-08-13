#include <string.h>
#include <connx/operator.h>
#include <connx/backend.h>

bool opset_Identity(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* output = CONNX_GET_OUTPUT(0);
	connx_Tensor* input = CONNX_GET_INPUT(0);

	// Create output if NULL
	if(output == NULL) {
		output = connx_Tensor_create(backend->pal, input->type, input->dimension, input->lengths);
		CONNX_SET_OUTPUT(0, output);
	}

	memcpy(output->base, input->base, connx_DataType_size(input->type) * connx_Tensor_total(input));
	return true;
}
