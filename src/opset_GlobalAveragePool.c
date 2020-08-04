#include <string.h>
#include <connx/operator.h>
#include <connx/backend.h>

bool opset_GlobalAveragePool(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	// outputs
	connx_Tensor* Y = CONNX_GET_OUTPUT(0);

	// inputs
	connx_Tensor* X = CONNX_GET_INPUT(0);

	if(Y == NULL) {
		uint32_t lengths[X->dimension];
		memcpy(lengths, X->lengths, sizeof(uint32_t) * 2);
		for(uint32_t i = 2; i < X->dimension; i++)
			lengths[i] = 1;

		Y = connx_Tensor_create(backend->hal, X->type, X->dimension, lengths);
		CONNX_SET_OUTPUT(0, Y);
	}

	uint32_t unit = 1;
	for(uint32_t i = 2; i < X->dimension; i++)
		unit *= X->lengths[i];

	switch(X->type) {
		case connx_FLOAT16:
		case connx_FLOAT32:
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
		case connx_FLOAT64:
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
			backend->hal->error(backend->hal, "Unsupported type: %d\n", X->type);
			return false;
	}

	return true;
}
