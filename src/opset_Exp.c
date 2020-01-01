#include <math.h>
#include <stdlib.h>
#include <connx/connx.h>

static bool Exp_resolve(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];

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

static bool Exp_exec(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];

	uint32_t total = connx_Tensor_total(X);

	switch(X->elemType) {
		case connx_DataType_FLOAT16:
			{
				float* x = (float*)X->base;
				float* y = (float*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = expf(x[i]);
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* x = (float*)X->base;
				float* y = (float*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = expf(x[i]);
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* x = (double*)X->base;
				double* y = (double*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = exp(x[i]);
				}
			}
			break;
		default:
			abort();
	}

	return true;
}

bool connx_opset_Exp_init() {
	connx_Operator_add("Exp", 1, 1, 0, Exp_resolve, Exp_exec,
		connx_DataType_TENSOR_FLOAT,
		connx_DataType_TENSOR_FLOAT);

	return true;
}
