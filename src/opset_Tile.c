#include <inttypes.h>
#include <string.h>
#include <strings.h>
#include <connx/operator.h>
#include <connx/backend.h>

static uint8_t* copy_leaf(uint8_t* output_base, uint8_t* input_base, uint32_t input_length, int64_t repeat, uint32_t elem_size) {
	uint32_t chunk = input_length * elem_size;

	switch(chunk) {
		case 1:
			for(uint32_t i = 0; i < repeat; i++) {
				*output_base = *input_base;
				output_base++;
			}
			break;
		case 2:
			for(uint32_t i = 0; i < repeat; i++) {
				*(uint16_t*)output_base = *(uint16_t*)input_base;
				output_base += 2;
			}
			break;
		case 4:
			for(uint32_t i = 0; i < repeat; i++) {
				*(uint32_t*)output_base = *(uint32_t*)input_base;
				output_base += 4;
			}
			break;
		case 8:
			for(uint32_t i = 0; i < repeat; i++) {
				*(uint64_t*)output_base = *(uint64_t*)input_base;
				output_base += 8;
			}
			break;
		default:
			for(uint32_t i = 0; i < repeat; i++) {
				memcpy(output_base, input_base, chunk);
				output_base += chunk;
			}
	}

	return output_base;
}

static uint8_t* copy_node(uint8_t* output_base, uint32_t* output_lengths, uint8_t* input_base, uint32_t* input_lengths, uint32_t idx, int64_t* repeats_base, uint32_t* units, uint32_t count, uint32_t elem_size) {
	if(idx + 1 < count) {
		for(uint32_t i = 0; i < repeats_base[idx]; i++) {
			for(uint32_t i = 0; i < *input_lengths; i++) {
				output_base = copy_node(output_base, output_lengths + 1, input_base + (i * units[idx] * elem_size), input_lengths + 1, idx + 1, repeats_base, units, count, elem_size);
			}
		}
	} else {
		output_base = copy_leaf(output_base, input_base, *(input_lengths), repeats_base[idx], elem_size);
	}

	return output_base;
}

bool opset_Tile(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* output = CONNX_GET_OUTPUT(0);
	connx_Tensor* input = CONNX_GET_INPUT(0);
	connx_Tensor* repeats = CONNX_GET_INPUT(1);

	uint32_t units[output->dimension];
	units[output->dimension - 1] = 1;
	for(uint32_t i = output->dimension - 1; i > 0; i--) {
		uint32_t idx = i - 1;
		units[idx] = units[idx + 1] * input->lengths[idx + 1];
	}

	copy_node(output->base, output->lengths, input->base, input->lengths, 0, (int64_t*)repeats->base, units, output->dimension, connx_DataType_size(output->type));

	return true;
}
