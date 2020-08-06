#include <inttypes.h>
#include <string.h>
#include <connx/operator.h>
#include <connx/backend.h>

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

static bool next_transposed(uint32_t* indices, uint32_t* lengths, uint32_t* perms, uint32_t dimension) {
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

bool opset_Transpose(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* output = CONNX_GET_OUTPUT(0);
	connx_Tensor* input = CONNX_GET_INPUT(0);
	connx_AttributeInts* perm = CONNX_GET_ATTRIBUTE(0);

	uint32_t dimension = input->dimension;

	// normalize perm
	uint32_t perm_values[output->dimension];

	if(perm != NULL) {
		memcpy(perm_values, perm->values, sizeof(int32_t) * dimension);
	} else {
		for(uint32_t i = 0; i < dimension; i++) {
			perm_values[i] = dimension - i - 1;
		}
	}

	// Create output if NULL
	if(output == NULL) {
		uint32_t lengths[dimension];
		for(uint32_t i = 0; i < input->dimension; i++) {
			lengths[i] = input->lengths[perm_values[i]];
		}

		output = connx_Tensor_create(backend->hal, input->type, input->dimension, lengths);
		CONNX_SET_OUTPUT(0, output);
	}


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

	switch(connx_DataType_size(output->type)) {
		case 1:
			{
				uint8_t* output_base = (uint8_t*)output->base;
				uint8_t* input_base = (uint8_t*)input->base;

				do {
					uint32_t output_idx = get_index(output_indices, output_units, output->dimension);
					uint32_t input_idx = get_index(input_indices, input_units, input->dimension);

					output_base[output_idx] = input_base[input_idx];
				} while(next(output_indices, output->lengths, output->dimension) && 
						next_transposed(input_indices, input->lengths, perm_values, input->dimension));
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
						next_transposed(input_indices, input->lengths, perm_values, input->dimension));
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
						next_transposed(input_indices, input->lengths, perm_values, input->dimension));
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
						next_transposed(input_indices, input->lengths, perm_values, input->dimension));
			}
			break;
		default:
			backend->hal->error(backend->hal, "Illegal element size: %" PRIu32 ", type: %" PRIu32, connx_DataType_size(output->type), output->type);
			return false;
	}

	return true;
}
