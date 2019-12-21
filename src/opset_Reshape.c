#include <string.h>
#include <connx/connx.h>

static bool Reshape_resolve(uintptr_t* stack) {
	return true;
}

static bool Reshape_exec(uintptr_t* stack) {
	connx_Tensor* reshaped = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];
	connx_Tensor* shape = (void*)stack[3];

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
				connx_exception("zero index cannot be set out of bounds: %ld", i);
				return false;
			}
			len *= lengths[i];
		} else if(s[i] < 0) {
			if(guess == -1) {
				guess = i;
			} else {
				connx_exception("-1 index cannot be repeated more than once: %ld", i);
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
		connx_exception("shape is not maching: data(%u) vs shape(%u)", total, len);
		return false;
	}

	memcpy(reshaped->base, data->base, connx_DataType_size(data->elemType) * len);

	return true;
}

bool connx_opset_Reshape_init() {
	connx_Operator_add("Reshape", 1, 2, 0, Reshape_resolve, Reshape_exec,
		connx_DataType_TENSOR_NUMBER,
		connx_DataType_TENSOR_NUMBER,
		connx_DataType_TENSOR_INT64);

	return true;
}
