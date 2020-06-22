#include <stdlib.h>
#include <connx/connx.h>

static bool LeakyRelu_resolve(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	__attribute__((unused)) void* attr_alpha = (void*)stack[3];

	// Create Y if NULL
	if(Y == NULL) {
		Y = connx_Tensor_create2(X->elemType, X->dimension, X->lengths);
		connx_Operator_stack_update(Y, 1, 1);
	}

	if(!connx_Tensor_isShapeEquals(X, Y)) {
		char buf1[32];
		char buf2[32];
		connx_Tensor_toShapeString(X, 32, buf1);
		connx_Tensor_toShapeString(Y, 32, buf2);
		connx_exception("X and Y's shape is different: %s vs %s", buf1, buf2);
		return false;
	}

	return true;
}

static bool LeakyRelu_exec(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	void* attr_alpha = (void*)stack[3];

	float* alpha = attr_alpha;

	uint32_t total = connx_Tensor_total(X);

	switch(X->elemType) {
		case connx_DataType_FLOAT16:
			{
				float* x = (float*)X->base;
				float* y = (float*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = x[i] > 0 ? x[i] : x[i] * *alpha;
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* x = (float*)X->base;
				float* y = (float*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = x[i] > 0 ? x[i] : x[i] * *alpha;
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* x = (double*)X->base;
				double* y = (double*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = x[i] > 0 ? x[i] : x[i] * *alpha;
				}
			}
			break;
		default:
			abort();
	}

	return true;
}

bool connx_opset_LeakyRelu_init() {
	connx_Operator_add("LeakyRelu", 1, 1, 1, LeakyRelu_resolve, LeakyRelu_exec,
		connx_DataType_TENSOR_FLOAT,
		connx_DataType_TENSOR_FLOAT,
		"alpha", connx_DataType_FLOAT32, 0.01);

	return true;
}
