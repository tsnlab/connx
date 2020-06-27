#include <inttypes.h>
#include <string.h>
#include <strings.h>
#include <connx/connx.h>

static bool Tile_resolve(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];
	connx_Tensor* repeats = (void*)stack[3];

	// input and output's element type is same
	if(output->elemType != input->elemType) {
		connx_exception("input and output's element type is differ: %" PRIu32 " != %" PRIu32, input->elemType, output->elemType);
		return false;
	}

	// Check repeat's dimension is 1
	if(repeats->dimension != 1) {
		connx_exception("repeats's dimension is not 1 but %" PRIu32, repeats->dimension);
		return false;
	}

	// Check repeat's lengths is same as input and output's
	if(repeats->lengths[0] != input->dimension) {
		connx_exception("input's dimension is differ from repeat's length: %" PRIu32 " != %" PRIu32, input->dimension, repeats->lengths[0]);
		return false;
	}

	if(repeats->lengths[0] != output->dimension) {
		connx_exception("output's dimension is differ from repeat's length: %" PRIu32 " != %" PRIu32, output->dimension, repeats->lengths[0]);
		return false;
	}

	// output's shape
	int64_t* repeats_base = (int64_t*)repeats->base;
	for(uint32_t i = 0; i < output->dimension; i++) {
		if(input->lengths[i] * repeats_base[i] != output->lengths[i]) {
			connx_exception("Illegal output's length[%" PRIu32 "] = %" PRIu32 ", expcted: %" PRIu32, i, output->lengths[i], input->lengths[i] * repeats_base[i]);
			return false;
		}
	}

	return true;
}

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

static bool Tile_exec(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];
	connx_Tensor* repeats = (void*)stack[3];

	uint32_t units[output->dimension];
	units[output->dimension - 1] = 1;
	for(uint32_t i = output->dimension - 1; i > 0; i--) {
		uint32_t idx = i - 1;
		units[idx] = units[idx + 1] * input->lengths[idx + 1];
	}

	copy_node(output->base, output->lengths, input->base, input->lengths, 0, (int64_t*)repeats->base, units, output->dimension, connx_DataType_size(output->elemType));

	return true;
}

bool connx_opset_Tile_init() {
	connx_Operator_add("Tile", 1, 2, 0, Tile_resolve, Tile_exec,
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// output
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// input
		connx_DataType_TENSOR_INT64);	// repeats

	return true;
}
