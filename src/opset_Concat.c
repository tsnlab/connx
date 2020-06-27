#include <inttypes.h>
#include <string.h>
#include <strings.h>
#include <connx/connx.h>

static bool Concat_resolve(uintptr_t* stack) {
	uint32_t input_count = connx_Stack_input_count(stack);

	connx_Tensor* concat_result = connx_Stack_outputs(stack)[0];
	connx_Tensor** inputs = (connx_Tensor**)connx_Stack_inputs(stack);
	int64_t* axis = connx_Stack_attributes(stack)[0];

	// Check axis
	if(*axis < 0)
		*axis += inputs[0]->dimension;

	if(*axis < 0 || *axis >= inputs[0]->dimension) {
		connx_exception("axis out of bounds: %" PRId64, *axis);
		return false;
	}

	// Create concat_result if NULL
	if(concat_result == NULL) {
		uint32_t lengths[inputs[0]->dimension];
		memcpy(lengths, inputs[0]->lengths, sizeof(uint32_t) * inputs[0]->dimension);

		for(uint32_t i = 1; i < input_count; i++) {
			lengths[i] += inputs[i]->lengths[*axis];
		}

		concat_result = connx_Tensor_create2(inputs[0]->elemType, inputs[0]->dimension, lengths);
		connx_Stack_update(1, concat_result);
	}

	// Check every input shape is same
	for(uint32_t i = 1; i < input_count; i++) {
		if(inputs[0]->elemType != inputs[i]->elemType) {
			connx_exception("input[0] and input[%" PRIu32 "]'s elemType is differ", i);
			return false;
		}

		if(inputs[0]->dimension != inputs[i]->dimension) {
			connx_exception("input[0] and input[%" PRIu32 "]'s dimension is differ", i);
			return false;
		}

		for(uint32_t j = 0; j < inputs[0]->dimension; j++) {
			if(j == *axis)
				continue;

			if(inputs[0]->lengths[j] != inputs[i]->lengths[j]) {
				connx_exception("input[0] and input[%" PRIu32 "]'s length[%" PRIu32 "] is differ: %" PRIu32 " != %" PRIu32, i, j, inputs[0]->lengths[j], inputs[i]->lengths[j]);
				return false;
			}
		}
	}

	// Check input and concat_result's element type
	if(inputs[0]->elemType != concat_result->elemType) {
		connx_exception("input and concat_result's element type is differ: %" PRIu32 " != %" PRIu32, inputs[0]->elemType, concat_result->elemType);
		return false;
	}

	// Check concat_result's dimension
	if(concat_result->dimension != inputs[0]->dimension) {
		connx_exception("input and concat_result's dimension is differ: %" PRIu32 " != %" PRIu32, inputs[0]->dimension, concat_result->dimension);
		return false;
	}

	// Check concat_result's shape
	for(uint32_t i = 0; i < concat_result->dimension; i++) {
		uint32_t expected;
		if(i == *axis) {
			expected = 0;
			for(uint32_t j = 0; j < input_count; j++) {
				expected += inputs[j]->lengths[i];
			}
		} else {
			expected = inputs[0]->lengths[i];
		}

		if(concat_result->lengths[i] != expected) {
			connx_exception("concat_result's length[%" PRIu32 "] is wrong: %" PRIu32 ", expected: %" PRIu32, i, concat_result->lengths[i], expected);
			return false;
		}
	}

	return true;
}

static bool Concat_exec(uintptr_t* stack) {
	uint32_t input_count = connx_Stack_input_count(stack);

	connx_Tensor* concat_result = connx_Stack_outputs(stack)[0];
	connx_Tensor** inputs = (connx_Tensor**)connx_Stack_inputs(stack);
	int64_t* axis = connx_Stack_attributes(stack)[0];

	// calculate batches
	uint32_t batches[input_count];
	for(uint32_t i = 0; i < input_count; i++) {
		batches[i] = 1;

		for(uint32_t j = *axis; j < inputs[0]->dimension; j++) {
			batches[i] *= inputs[i]->lengths[j];
		}
	}

	// calculate units
	uint32_t units[input_count][*axis];
	if(*axis > 0) {
		for(uint32_t i = 0; i < input_count; i++) {
			units[i][*axis - 1] = batches[i];

			for(uint32_t j = *axis; j > 1; j--) {
				units[i][j - 2] = units[i][j - 1] * inputs[i]->lengths[j - 1];
			}
		}
	}

	// init indices
	uint32_t indices[*axis];
	bzero(indices, *axis * sizeof(uint32_t));
	uint32_t elem_size = connx_DataType_size(concat_result->elemType);
	uint8_t* concat_result_base = concat_result->base;
	uint32_t chunks[input_count];
	for(uint32_t i = 0; i < input_count; i++)
		chunks[i] = batches[i] * elem_size;

	while(true) {
		// batch copying
		for(uint32_t i = 0; i < input_count; i++) {
			// calculating index
			uint32_t idx = 0;
			for(uint32_t j = 0; j < *axis; j++) {
				idx += indices[j] * units[i][j];
			}

			uint8_t* input_base = inputs[i]->base;

			switch(chunks[i]) {
				case 1:
					*concat_result_base++ = input_base[idx * elem_size];
					break;
				case 2:
					*(uint16_t*)concat_result_base = *(uint16_t*)(input_base + idx * elem_size);
					concat_result_base += 2;
					break;
				case 4:
					*(uint32_t*)concat_result_base = *(uint32_t*)(input_base + idx * elem_size);
					concat_result_base += 4;
					break;
				case 8:
					*(uint64_t*)concat_result_base = *(uint64_t*)(input_base + idx * elem_size);
					concat_result_base += 8;
					break;
				default:
					memcpy(concat_result_base, input_base + idx * elem_size, batches[i] * elem_size);
					concat_result_base += batches[i] * elem_size;
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
