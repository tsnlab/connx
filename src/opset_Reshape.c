#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <connx/operator.h>
#include <connx/backend.h>

bool opset_Reshape(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* reshaped = CONNX_GET_OUTPUT(0);
	connx_Tensor* data = CONNX_GET_INPUT(0);
	connx_Tensor* shape = CONNX_GET_INPUT(1);

	uint32_t dimension = 1;
	for(uint32_t i = 0; i < shape->dimension; i++) {
		dimension *= shape->lengths[i];
	}

	uint32_t len = 1;
	int32_t guess = -1;
	uint32_t lengths[dimension];
	int64_t* s = (int64_t*)shape->base;
	for(uint32_t i = 0; i < dimension; i++) {
		if(s[i] == 0) {
			if(i < data->dimension) {
				lengths[i] = data->lengths[i];
			} else {
				backend->hal->error(backend->hal, "zero index cannot be set out of bounds: %" PRIu32 " >= %" PRIu32 "\n", i, data->dimension);
				return false;
			}
			len *= lengths[i];
		} else if(s[i] < 0) {
			if(guess == -1) {
				guess = i;
			} else {
				backend->hal->error(backend->hal, "-1 index cannot be repeated more than once: %" PRIu32 "\n", i);
				return false;
			}
		} else {
			lengths[i] = s[i];
			len *= lengths[i];
		}
	}

	uint32_t total = connx_Tensor_total(data);

	if(guess >= 0) {
		lengths[guess] = total / len;
		len = total;
	}

	if(total != len) {
		backend->hal->error(backend->hal, "shape is not maching: data(%" PRIu32 ") vs shape(%" PRIu32 ")\n", total, len);
		return false;
	}

	if(reshaped == NULL) {
		reshaped = connx_Tensor_create(backend->hal, data->type, dimension, lengths);
		CONNX_SET_OUTPUT(0, reshaped);
	} else {
		backend->hal->free(backend->hal, reshaped->lengths);
		reshaped->dimension = dimension;
		reshaped->lengths = backend->hal->alloc(backend->hal, sizeof(uint32_t) * dimension);
		memcpy(reshaped->lengths, lengths, sizeof(uint32_t) * dimension);
	}

	memcpy(reshaped->base, data->base, connx_DataType_size(data->type) * len);

	return true;
}
