#include <inttypes.h>
#include <connx/connx.h>

static bool Transpose_resolve(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];
	void* attr_perm = (void*)stack[3];

	int64_t* perm = connx_Attribute_base(attr_perm);
	uint32_t perm_length = connx_Attribute_length(attr_perm);

	if(perm_length != 0) {
		if(input->dimension != perm_length) {
			connx_exception("Illegal perm length: %" PRIu32 ", must be %" PRIu32, perm_length, input->dimension);
			return false;
		}

		for(uint32_t i = 0; i < perm_length; i++) {
			for(uint32_t j = 0; j < perm_length; j++) {
				if(perm[j] == i)
					goto found;
			}

			connx_exception("index %" PRIu32 " is not found in perm the attrbitue", i);
			return false;

found:
			continue;
		}
	} else {
		int64_t array[input->dimension];
		for(uint32_t i = 0; i < input->dimension; i++) {
			array[i] = input->dimension - i - 1;
		}

		connx_Attribute_delete(attr_perm);
		stack[3] = connx_Attribute_create_ints(input->dimension, array);
		attr_perm = (void*)stack[3];

		perm = connx_Attribute_base(attr_perm);
		connx_Attribute_length(attr_perm);
	}

	// Create output if NULL
	if(output == NULL) {
		uint32_t lengths[input->dimension];
		for(uint32_t i = 0; i < input->dimension; i++) {
			lengths[i] = input->lengths[perm[i]];
		}

		output = connx_Tensor_create2(input->elemType, input->dimension, lengths);
		connx_Stack_update(1, output);
	}

	if(output->elemType != input->elemType) {
		connx_exception("Input and output's element type is differ: %" PRIu32 " vs %" PRIu32, input->elemType, output->elemType);
		return false;
	}

	if(connx_Tensor_total(output) != connx_Tensor_total(input)) {
		connx_exception("Input and output's total length is differ: %" PRIu32 " vs %" PRIu32, 
			connx_Tensor_total(input), connx_Tensor_total(output));
		return false;
	}

	return true;
}

static bool next(uint32_t* indices, uint32_t* lengths, uint32_t dimension) {
	for(uint32_t i = 0; i < dimension; i++) {
		uint32_t bi = dimension - i - 1;

		if(++indices[bi] < lengths[bi])
			goto next;

		indices[bi] = 0;
	}

	return false;

next:
	return true;
}

static bool next_transposed(uint32_t* indices, uint32_t* lengths, int64_t* perms, uint32_t dimension) {
	for(uint32_t i = 0; i < dimension; i++) {
		uint32_t bi = dimension - i - 1;

		if(++indices[perms[bi]] < lengths[perms[bi]])
			goto next;

		indices[perms[bi]] = 0;
	}

	return false;

next:
	return true;
}

static uint32_t get_index(uint32_t* indices, uint32_t* units, uint32_t dimension) {
	uint32_t idx = 0;
	for(uint32_t i = 0; i < dimension; i++) {
		idx += indices[i] * units[i];
	}

	return idx;
}

static bool Transpose_exec(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];
	void* attr_perm = (void*)stack[3];

	int64_t* perm = connx_Attribute_base(attr_perm);
	__attribute__((unused)) uint32_t perm_length = connx_Attribute_length(attr_perm);

#define CONNX_MAX_INDICES	16

	uint32_t output_indices[CONNX_MAX_INDICES] = { 0, };
	uint32_t output_units[CONNX_MAX_INDICES] = { 0, };

	for(uint32_t i = 0; i < output->dimension; i++) {
		uint32_t bi = output->dimension - i - 1;
		if(i == 0) {
			output_units[bi] = 1;
		} else {
			output_units[bi] = output_units[bi + 1] * output->lengths[bi + 1];
		}
	}

	uint32_t input_indices[CONNX_MAX_INDICES] = { 0, };
	uint32_t input_units[CONNX_MAX_INDICES] = { 0, };

	for(uint32_t i = 0; i < input->dimension; i++) {
		uint32_t bi = input->dimension - i - 1;
		if(i == 0) {
			input_units[bi] = 1;
		} else {
			input_units[bi] = input_units[bi + 1] * input->lengths[bi + 1];
		}
	}

	switch(connx_DataType_size(output->elemType)) {
		case 1:
			{
				uint8_t* output_base = (uint8_t*)output->base;
				uint8_t* input_base = (uint8_t*)input->base;

				do {
					uint32_t output_idx = get_index(output_indices, output_units, output->dimension);
					uint32_t input_idx = get_index(input_indices, input_units, input->dimension);

					output_base[output_idx] = input_base[input_idx];
				} while(next(output_indices, output->lengths, output->dimension) && 
						next_transposed(input_indices, input->lengths, perm, input->dimension));
			}
			break;
		case 2:
			{
				uint16_t* output_base = (uint16_t*)output->base;
				uint16_t* input_base = (uint16_t*)input->base;

				do {
					uint32_t output_idx = get_index(output_indices, output_units, output->dimension);
					uint32_t input_idx = get_index(input_indices, input_units, input->dimension);

					output_base[output_idx] = input_base[input_idx];
				} while(next(output_indices, output->lengths, output->dimension) && 
						next_transposed(input_indices, input->lengths, perm, input->dimension));
			}
			break;
		case 4:
			{
				uint32_t* output_base = (uint32_t*)output->base;
				uint32_t* input_base = (uint32_t*)input->base;

				do {
					uint32_t output_idx = get_index(output_indices, output_units, output->dimension);
					uint32_t input_idx = get_index(input_indices, input_units, input->dimension);

					output_base[output_idx] = input_base[input_idx];
				} while(next(output_indices, output->lengths, output->dimension) && 
						next_transposed(input_indices, input->lengths, perm, input->dimension));
			}
			break;
		case 8:
			{
				uint64_t* output_base = (uint64_t*)output->base;
				uint64_t* input_base = (uint64_t*)input->base;

				do {
					uint32_t output_idx = get_index(output_indices, output_units, output->dimension);
					uint32_t input_idx = get_index(input_indices, input_units, input->dimension);

					output_base[output_idx] = input_base[input_idx];
				} while(next(output_indices, output->lengths, output->dimension) && 
						next_transposed(input_indices, input->lengths, perm, input->dimension));
			}
			break;
		default:
			connx_exception("Illegal element size: %" PRIu32 ", type: %" PRIu32, connx_DataType_size(output->elemType), output->elemType);
			return false;
	}

	return true;
}

bool connx_opset_Transpose_init() {
	connx_Operator_add("Transpose", 1, 1, 1, Transpose_resolve, Transpose_exec,
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,
		"perm", connx_DataType_INT64_ARRAY, 0, NULL);

	return true;
}
