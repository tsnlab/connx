#include <math.h>
#include <stdlib.h>
#include <connx/connx.h>

static bool Round_resolve(uintptr_t* stack) {
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

static bool Round_exec(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];

	uint32_t total = connx_Tensor_total(X);

	switch(X->elemType) {
		case connx_DataType_FLOAT16:
			{
				float* x = (float*)X->base;
				float* y = (float*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = roundf(x[i]);

					float diff = y[i] - x[i];
					int mod = (int)y[i] % 2;
					if(mod == 1 && diff == 0.5f) {
						y[i]--;
					} else if(mod == -1 && diff == -0.5f) {
						y[i]++;
					}
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* x = (float*)X->base;
				float* y = (float*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = roundf(x[i]);

					float diff = y[i] - x[i];
					int mod = (int)y[i] % 2;
					if(mod == 1 && diff == 0.5f) {
						y[i]--;
					} else if(mod == -1 && diff == -0.5f) {
						y[i]++;
					}
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* x = (double*)X->base;
				double* y = (double*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = round(x[i]);

					double diff = y[i] - x[i];
					int mod = (int)y[i] % 2;
					if(mod == 1 && diff == 0.5) {
						y[i]--;
					} else if(mod == -1 && diff == -0.5) {
						y[i]++;
					}
				}
			}
			break;
		default:
			abort();
	}

	return true;
}

bool connx_opset_Round_init() {
	connx_Operator_add("Round", 1, 1, 0, Round_resolve, Round_exec,
		connx_DataType_TENSOR_FLOAT,
		connx_DataType_TENSOR_FLOAT);

	return true;
}
