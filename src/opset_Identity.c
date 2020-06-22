#include <connx/connx.h>

static bool Identity_resolve(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];

	// Create output if NULL
	if(output == NULL) {
		output = connx_Tensor_create2(input->elemType, input->dimension, input->lengths);
		connx_Operator_stack_update(output, 1, 1);
	}

	if(!connx_Tensor_isShapeEquals(output, input)) {
		connx_exception("Input and output's shape is not equal\n");
		return false;
	}

	return true;
}

static bool Identity_exec(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];

	return connx_Tensor_copy(input, output);
}

bool connx_opset_Identity_init() {
	connx_Operator_add("Identity", 1, 1, 0, Identity_resolve, Identity_exec,
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL);

	return true;
}
