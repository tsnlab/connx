#include <inttypes.h>
#include <string.h>
#include <strings.h>
#include <connx/operator.h>
#include <connx/backend.h>

bool opset_Concat(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	uint32_t input_count = CONNX_INPUT_COUNT(counts);

	connx_Tensor* concat_result = CONNX_GET_OUTPUT(0);
	connx_Tensor* inputs[input_count];
	for(uint32_t i = 0; i < input_count; i++) {
		inputs[i] = CONNX_GET_INPUT(i);
	}

	connx_AttributeInt* axis = CONNX_GET_ATTRIBUTE(0);

	// normalize axis
	uint32_t axis_value = axis->value < 0 ? axis->value + inputs[0]->dimension : axis->value;

	// Create concat_result if NULL
	if(concat_result == NULL) {
		uint32_t lengths[inputs[0]->dimension];
		memcpy(lengths, inputs[0]->lengths, sizeof(uint32_t) * inputs[0]->dimension);

		for(uint32_t i = 1; i < input_count; i++) {
			lengths[i] += inputs[i]->lengths[axis_value];
		}

		concat_result = connx_Tensor_create(backend->hal, inputs[0]->type, inputs[0]->dimension, lengths);
		CONNX_SET_OUTPUT(0, concat_result);
	}

	// calculate batches
	uint32_t batches[input_count];
	for(uint32_t i = 0; i < input_count; i++) {
		batches[i] = 1;

		for(uint32_t j = axis_value; j < inputs[0]->dimension; j++) {
			batches[i] *= inputs[i]->lengths[j];
		}
	}

	// calculate units
	uint32_t units[input_count][axis_value];
	if(axis_value > 0) {
		for(uint32_t i = 0; i < input_count; i++) {
			units[i][axis_value - 1] = batches[i];

			for(uint32_t j = axis_value; j > 1; j--) {
				units[i][j - 2] = units[i][j - 1] * inputs[i]->lengths[j - 1];
			}
		}
	}

	// init indices
	uint32_t indices[axis_value];
	bzero(indices, axis_value * sizeof(uint32_t));
	uint32_t elem_size = connx_DataType_size(concat_result->type);
	uint8_t* concat_result_base = concat_result->base;
	uint32_t chunks[input_count];
	for(uint32_t i = 0; i < input_count; i++)
		chunks[i] = batches[i] * elem_size;

	while(true) {
		// batch copying
		for(uint32_t i = 0; i < input_count; i++) {
			// calculating index
			uint32_t idx = 0;
			for(uint32_t j = 0; j < axis_value; j++) {
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
		for(uint32_t i = 0; i < axis_value; i++) {
			uint32_t idx = axis_value - i - 1;

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
