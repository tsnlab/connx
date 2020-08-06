#include <inttypes.h>
#include <string.h>
#include <strings.h>
#include <connx/operator.h>
#include <connx/backend.h>

bool opset_Slice(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* output = CONNX_GET_OUTPUT(0);
	connx_Tensor* data = CONNX_GET_INPUT(0);
	connx_Tensor* starts = CONNX_GET_INPUT(1);
	connx_Tensor* ends = CONNX_GET_INPUT(2);
	connx_Tensor* axes = CONNX_GET_INPUT(3);
	connx_Tensor* steps = CONNX_GET_INPUT(4);

	uint32_t dimension = data->dimension;

	// normalize starts
	int32_t starts_base[dimension];
	switch(starts->type) {
		case connx_INT32:
			memcpy(starts_base, starts->base, sizeof(int32_t) * dimension);
			break;
		case connx_INT64: {
				int64_t* array = (int64_t*)starts->base;

				for(uint32_t i = 0; i < dimension; i++) {
					starts_base[i] = array[i];
				}
			}
			break;
		default:
			backend->hal->error(backend->hal, "Illegal starts type: %d\n", starts->type);
			return false;
	}

	// normalize ends
	int32_t ends_base[dimension];
	switch(ends->type) {
		case connx_INT32:
			memcpy(ends_base, ends->base, sizeof(int32_t) * dimension);
			break;
		case connx_INT64: {
				int64_t* array = (int64_t*)ends->base;

				for(uint32_t i = 0; i < dimension; i++) {
					ends_base[i] = array[i];
				}
			}
			break;
		default:
			backend->hal->error(backend->hal, "Illegal ends type: %d\n", ends->type);
			return false;
	}

	// normalize steps
	int32_t steps_base[dimension];
	if(steps != NULL) {
		switch(steps->type) {
			case connx_INT32:
				memcpy(steps_base, steps->base, sizeof(int32_t) * dimension);
				break;
			case connx_INT64: {
					int64_t* array = (int64_t*)steps->base;

					for(uint32_t i = 0; i < dimension; i++) {
						steps_base[i] = array[i];
					}
				}
				break;
			default:
				backend->hal->error(backend->hal, "Illegal steps type: %d\n", steps->type);
				return false;
		}
	} else {
		for(uint32_t i = 0; i < dimension; i++) {
			steps_base[i] = 1;
		}
	}

	// axes
	if(axes != NULL) {
		int32_t old_starts_base[dimension];
		memcpy(old_starts_base, starts_base, sizeof(int32_t) * dimension);

		int32_t old_ends_base[dimension];
		memcpy(old_ends_base, ends_base, sizeof(int32_t) * dimension);

		int32_t old_steps_base[dimension];
		memcpy(old_steps_base, steps_base, sizeof(int32_t) * dimension);

		int32_t* axes32_base = (int32_t*)axes->base;
		int64_t* axes64_base = (int64_t*)axes->base;
		for(uint32_t i = 0; i < dimension; i++) {
			int32_t index = 0;

			switch(axes->type) {
				case connx_INT32:
					index = axes32_base[i];
					break;
				case connx_INT64:
					index = axes64_base[i];
					break;
				default:
					backend->hal->error(backend->hal, "Illegal axes type: %d\n", axes->type);
					return false;
			}

			if(index < 0)
				index += dimension;

			starts_base[i] = old_starts_base[index];
			ends_base[i] = old_ends_base[index];
			steps_base[i] = old_steps_base[index];
		}
	}

	// Create output if NULL
	if(output == NULL) {
		uint32_t lengths[data->dimension];

		for(uint32_t i = 0; i < data->dimension; i++) {
			int32_t len = ends_base[i] - starts_base[i];
			int32_t rem = len % steps_base[i] == 0 ? 0 : 1;
			len /= steps_base[i];
			len += rem;

			lengths[i] = len;
		}

		output = connx_Tensor_create(backend->hal, data->type, data->dimension, lengths);
		CONNX_SET_OUTPUT(0, output);
	}

	// Check empty output
	for(uint32_t i = 0; i < output->dimension; i++) {
		if(steps_base[i] > 0) {
			if(starts_base[i] >= ends_base[i]) {
				return true;
			}
		} else {
			if(starts_base[i] <= ends_base[i]) {
				return true;
			}
		}
	}

	// calculate batch and loop dimension(count)
	uint32_t batch = 1;
	uint32_t count = output->dimension;
	for(uint32_t i = 0; i < output->dimension; i++) {
		uint32_t idx = output->dimension - i - 1;

		if((steps_base[idx] == 1 && starts_base[idx] == 0 && ends_base[idx] == (int32_t)data->lengths[idx])) {
			batch *= (int32_t)data->lengths[idx];
			count--;
		} else {
			break;
		}
	}

	// calculates units
	uint32_t* lengths = data->lengths;
	uint32_t units[count];
	if(count > 0) {
		units[count - 1] = batch;

		for(uint32_t i = 0; i < count - 1; i++) {
			uint32_t idx = count - i - 2;

			units[idx] = units[idx + 1] * lengths[idx + 1];
		}
	}

	// init indices
	int32_t indices[count];
	for(uint32_t i = 0; i < count; i++) {
		indices[i] = starts_base[i];
	}

	uint32_t elem_size = connx_DataType_size(output->type);
	uint8_t* data_base = data->base;
	uint8_t* output_base = output->base;

	uint32_t chunk = batch * elem_size;
	while(true) {
		// calculating index
		uint32_t idx = 0;
		for(uint32_t i = 0; i < count; i++) {
			idx += indices[i] * units[i];
		}

		// batch copying
		switch(chunk) {
			case 1:
				*output_base++ = data_base[idx];
				break;
			case 2:
				*(uint16_t*)output_base = ((uint16_t*)data_base)[idx];
				output_base += 2;
				break;
			case 4:
				*(uint32_t*)output_base = ((uint32_t*)data_base)[idx];
				output_base += 4;
				break;
			case 8:
				*(uint64_t*)output_base = ((uint64_t*)data_base)[idx];
				output_base += 8;
				break;
			default:
				memcpy(output_base, data_base + idx * elem_size, batch * elem_size);
				output_base += batch * elem_size;
		}

		// step next
		for(uint32_t i = 0; i < count; i++) {
			uint32_t idx = count - i - 1;

			indices[idx] += steps_base[idx];

			if(steps_base[idx] > 0) {
				if(indices[idx] < ends_base[idx]) {
					goto next;
				}
			} else {
				if(indices[idx] > ends_base[idx]) {
					goto next;
				}
			}

			indices[idx] = starts_base[idx];
		}
		break;
next:
		;
	}

	return true;
}
