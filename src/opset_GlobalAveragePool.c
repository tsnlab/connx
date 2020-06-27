#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <connx/connx.h>

static bool GlobalAveragePool_resolve(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];

	// Create Y if NULL
	if(Y == NULL) {
		uint32_t lengths[X->dimension];
		memcpy(lengths, X->lengths, sizeof(uint32_t) * 2);
		for(uint32_t i = 2; i < X->dimension; i++)
			lengths[i] = 1;

		Y = connx_Tensor_create2(X->elemType, X->dimension, lengths);
		connx_Stack_update(1, Y);
	}

	if(X->elemType != Y->elemType) {
		connx_exception("X and Y's elemType is differ: %" PRIu32 " != %" PRIu32, X->elemType, Y->elemType);
		return false;
	}

	if(X->dimension != Y->dimension) {
		connx_exception("Illegal X's dimension is differ from Y: %" PRIu32 " != %" PRIu32, X->dimension, Y->dimension);
		return false;
	}

	return true;
}

static bool GlobalAveragePool_exec(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];

	uint32_t unit = 1;
	for(uint32_t i = 2; i < X->dimension; i++)
		unit *= X->lengths[i];

	switch(X->elemType) {
		case connx_DataType_FLOAT16:
		case connx_DataType_FLOAT32:
			{
				float* Y_base = (float*)Y->base;
				float* X_base = (float*)X->base;

				for(uint32_t batch = 0; batch < X->lengths[0]; batch++) {
					for(uint32_t channel = 0; channel < X->lengths[1]; channel++) {
						float average = 0;

						for(uint32_t i = 0; i < unit; i++) {
							average += (*X_base++) / unit;
						}

						*Y_base++ = average;
					}
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* Y_base = (double*)Y->base;
				double* X_base = (double*)X->base;

				for(uint32_t batch = 0; batch < X->lengths[0]; batch++) {
					for(uint32_t channel = 0; channel < X->lengths[1]; channel++) {
						double average = 0;

						for(uint32_t i = 0; i < unit; i++) {
							average += (*X_base++) / unit;
						}

						*Y_base++ = average;
					}
				}
			}
			break;
		default:
			abort();
	}

	return true;
}

bool connx_opset_GlobalAveragePool_init() {
	connx_Operator_add("GlobalAveragePool", 1, 1, 0, GlobalAveragePool_resolve, GlobalAveragePool_exec,
		connx_DataType_TENSOR_FLOAT,
		connx_DataType_TENSOR_FLOAT);

	return true;
}
