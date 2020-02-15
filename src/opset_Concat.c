#include <string.h>
#include <strings.h>
#include <connx/connx.h>

static bool Concat_resolve(uintptr_t* stack) {
	uintptr_t count = stack[0];

	connx_Tensor* concat_result = (void*)stack[1];
	uint32_t input_count = count - 2;

	connx_Tensor* inputs[input_count];
	for(uint32_t i = 0; i < input_count; i++)
		inputs[i] = (void*)stack[2 + i];

	int64_t* axis = (void*)stack[count];

	// Check every input shape is same
	for(uint32_t i = 1; i < input_count; i++) {
		if(!connx_Tensor_isShapeEquals(inputs[0], inputs[i])) {
			connx_exception("input[0] and input[%u]'s shape is differ", i);
			return false;
		}
	}

	// Check input and concat_result's element type
	if(inputs[0]->elemType != concat_result->elemType) {
		connx_exception("input and concat_result's element type is differ: %u != %u", inputs[0]->elemType, concat_result->elemType);
		return false;
	}

	// Check concat_result's dimension
	if(concat_result->dimension != inputs[0]->dimension) {
		connx_exception("input and concat_result's dimension is differ: %u != %u", inputs[0]->dimension, concat_result->dimension);
		return false;
	}

	// Check axis
	if(*axis < 0)
		*axis += concat_result->dimension;

	if(*axis < 0 || *axis >= concat_result->dimension) {
		connx_exception("axis out of bounds: %ld", *axis);
		return false;
	}

	// Check concat_result's shape
	for(uint32_t i = 0; i < concat_result->dimension; i++) {
		uint32_t expected;
		if(i == *axis) {
			expected = inputs[0]->lengths[i] * input_count;
		} else {
			expected = inputs[0]->lengths[i];
		}

		if(concat_result->lengths[i] != expected) {
			connx_exception("concat_result's length[%u] is wrong: %u, expected: %u", i, concat_result->lengths[i], expected);
			return false;
		}
	}

	return true;
}

static bool Concat_exec(uintptr_t* stack) {
	uintptr_t count = stack[0];

	connx_Tensor* concat_result = (void*)stack[1];
	uint32_t input_count = count - 2;

	connx_Tensor* inputs[input_count];
	for(uint32_t i = 0; i < input_count; i++)
		inputs[i] = (void*)stack[2 + i];

	int64_t* axis = (void*)stack[count];

	// calculate batch
	uint32_t batch = 1;
	for(uint32_t i = *axis; i < inputs[0]->dimension; i++) {
		batch *= inputs[0]->lengths[i];
	}

	// calculate units
	uint32_t units[*axis];
	if(*axis > 0) {
		units[*axis - 1] = batch;

		for(uint32_t i = *axis; i > 1; i--) {
			units[i - 2] = units[i - 1] * inputs[0]->lengths[i - 1];
		}
	}

	// init indices
	uint32_t indices[*axis];
	bzero(indices, *axis * sizeof(uint32_t));
	uint32_t elem_size = connx_DataType_size(concat_result->elemType);
	uint8_t* concat_result_base = concat_result->base;
	uint32_t chunk = batch * elem_size;

	while(true) {
		// calculating index
		uint32_t idx = 0;
		for(uint32_t i = 0; i < *axis; i++) {
			idx += indices[i] * units[i];
		}

		// batch copying
		switch(chunk) {
			case 1:
				for(uint32_t j = 0; j < input_count; j++) {
					uint8_t* input_base = inputs[j]->base;

					*concat_result_base++ = input_base[idx * elem_size];
				}
				break;
			case 2:
				for(uint32_t j = 0; j < input_count; j++) {
					uint8_t* input_base = inputs[j]->base;

					*(uint16_t*)concat_result_base = *(uint16_t*)(input_base + idx * elem_size);
					concat_result_base += 2;
				}
				break;
			case 4:
				for(uint32_t j = 0; j < input_count; j++) {
					uint8_t* input_base = inputs[j]->base;

					*(uint32_t*)concat_result_base = *(uint32_t*)(input_base + idx * elem_size);
					concat_result_base += 4;
				}
				break;
			case 8:
				for(uint32_t j = 0; j < input_count; j++) {
					uint8_t* input_base = inputs[j]->base;

					*(uint64_t*)concat_result_base = *(uint64_t*)(input_base + idx * elem_size);
					concat_result_base += 8;
				}
				break;
			default:
				for(uint32_t j = 0; j < input_count; j++) {
					uint8_t* input_base = inputs[j]->base;

					memcpy(concat_result_base, input_base + idx * elem_size, batch * elem_size);
					concat_result_base += batch * elem_size;
				}
		}

		// step next
		for(uint32_t i = 0; i < *axis; i++) {
			uint32_t idx = *axis - i - 1;

			if(++indices[idx] < inputs[0]->lengths[i])
				goto next;

			indices[idx] = 0;
		}
		break;
next:
		;
	}

	return true;
}

bool connx_opset_Concat_init() {
	connx_Operator_add("Concat", 1,  1 | CONNX_VARARGS, 1, Concat_resolve, Concat_exec,
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// concat_result
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// inputs
		"axis", connx_DataType_INT64, 0);

	return true;
}
