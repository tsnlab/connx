#include <math.h>
#include <stdlib.h>
#include <connx/connx.h>

static bool Ceil_resolve(uintptr_t* stack) {
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

static bool Ceil_exec(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];

	uint32_t total = connx_Tensor_total(X);

	switch(X->elemType) {
		case connx_DataType_FLOAT16:
			{
				float* x = (float*)X->base;
				float* y = (float*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = ceilf(x[i]);
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* x = (float*)X->base;
				float* y = (float*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = ceilf(x[i]);
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* x = (double*)X->base;
				double* y = (double*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = ceil(x[i]);
				}
			}
			break;
		default:
			abort();
	}

	return true;
}

bool connx_opset_Ceil_init() {
	connx_Operator_add("Ceil", 1, 1, 0, Ceil_resolve, Ceil_exec,
		connx_DataType_TENSOR_FLOAT,
		connx_DataType_TENSOR_FLOAT);

	return true;
}
