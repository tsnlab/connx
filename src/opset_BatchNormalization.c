#include <math.h>
#include <connx/operator.h>
#include <connx/backend.h>

bool opset_BatchNormalization(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	// outputs
	connx_Tensor* Y = CONNX_GET_OUTPUT(0);

	// inputs
	connx_Tensor* X = CONNX_GET_INPUT(0);
	connx_Tensor* scale = CONNX_GET_INPUT(1);
	connx_Tensor* B = CONNX_GET_INPUT(2);
	connx_Tensor* mean = CONNX_GET_INPUT(3);
	connx_Tensor* var = CONNX_GET_INPUT(4);

	// attributes
	connx_AttributeFloat* epsilon = CONNX_GET_ATTRIBUTE(0);
	__attribute__((unused)) connx_AttributeFloat* momentum = CONNX_GET_ATTRIBUTE(1);

	if(Y == NULL) {
		Y = connx_Tensor_create(backend->hal, X->type, X->dimension, X->lengths);
		CONNX_SET_OUTPUT(0, Y);
	}

	uint32_t batch_count = Y->lengths[0];
	uint32_t channel_count = Y->lengths[1];
	uint32_t unit_count = 1;
	for(uint32_t i = 2; i < Y->dimension; i++)
		unit_count *= Y->lengths[i];

	switch(Y->type) {
		case connx_FLOAT16:
		case connx_FLOAT32:
			{
				float* Y_base = (float*)Y->base;
				float* X_base = (float*)X->base;

				float* scales = (float*)scale->base;
				float* Bs = (float*)B->base;
				float* means = (float*)mean->base;
				float* vars = (float*)var->base;

				for(uint32_t b = 0; b < batch_count; b++) {
					for(uint32_t c = 0; c < channel_count; c++) {
						float scale_value = scales[c];
						float B_value = Bs[c];
						float mean_value = means[c];// * *momentum + means[c] * (1 - *momentum); // momentum is only applied on trainning
						float sqrt_value = sqrtf(vars[c] + epsilon->value);
						float scale_div_sqrt_value = scale_value / sqrt_value;

						for(uint32_t u = 0; u < unit_count; u++) {
							*Y_base++ = (*X_base++ - mean_value) * scale_div_sqrt_value + B_value;
						}
					}
				}
			}
			break;
		case connx_FLOAT64:
			{
				double* Y_base = (double*)Y->base;
				double* X_base = (double*)X->base;

				double* scales = (double*)scale->base;
				double* Bs = (double*)B->base;
				double* means = (double*)mean->base;
				double* vars = (double*)var->base;

				for(uint32_t b = 0; b < batch_count; b++) {
					for(uint32_t c = 0; c < channel_count; c++) {
						double scale_value = scales[c];
						double B_value = Bs[c];
						double mean_value = means[c];// * *momentum + means[c] * (1 - *momentum); // momentum is only applied on trainning
						double sqrt_value = sqrt(vars[c] + epsilon->value);
						double scale_div_sqrt_value = scale_value / sqrt_value;

						for(uint32_t u = 0; u < unit_count; u++) {
							*Y_base++ = (*X_base++ - mean_value) * scale_div_sqrt_value + B_value;
						}
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
