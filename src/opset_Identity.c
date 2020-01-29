#include <stdlib.h>
#include "opset.h"
#include <connx/connx.h>

#define MINMAX(min, max, a, b)	\
	if((a) > (b)) {		\
		(max) = (a); 	\
		(min) = (b);	\
	} else {			\
		(min) = (b);	\
		(max) = (a);	\
	}

static bool Identity_resolve(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];

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
