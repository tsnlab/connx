#include <inttypes.h>
#include <string.h>
#include <strings.h>
#include <connx/connx.h>

static bool Slice_resolve(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];
	connx_Tensor* starts = (void*)stack[3];
	connx_Tensor* ends = (void*)stack[4];
	connx_Tensor* axes = (void*)stack[5];
	connx_Tensor* steps = (void*)stack[6];

	// check starts dimension
	if(starts->dimension != 1) {
		connx_exception("starts' dimension is not 1 but %" PRIu32, starts->dimension);
		return false;
	}

	// check ends dimension
	if(ends->dimension != 1) {
		connx_exception("ends' dimension is not 1 but %" PRIu32, ends->dimension);
		return false;
	}

	// check steps dimension
	if(steps != NULL && steps->dimension != 1) {
		connx_exception("steps' dimension is not 1 but %" PRIu32, steps->dimension);
		return false;
	}

	// check axes dimension
	uint32_t axes_length = 0;
	int32_t axes_array[data->dimension];

	if(axes != NULL) {
		if(axes->dimension != 1) {
			connx_exception("axes' dimension is not 1 but %" PRIu32, axes->dimension);
			return false;
		} else {
			if(axes->lengths[0] != starts->lengths[0]) {
				connx_exception("axes' length and starts' length is differ: %" PRIu32 " vs %" PRIu32, axes->lengths[0], starts->lengths[0]);
				return false;
			}

			if(axes->lengths[0] != ends->lengths[0]) {
				connx_exception("axes' length and ends' length is differ: %" PRIu32 " vs %" PRIu32, axes->lengths[0], ends->lengths[0]);
				return false;
			}

			if(steps != NULL && axes->lengths[0] != steps->lengths[0]) {
				connx_exception("axes' length and steps' length is differ: %" PRIu32 " vs %" PRIu32, axes->lengths[0], steps->lengths[0]);
				return false;
			}
		}

		// normalize axes type to int32 array
		axes_length = axes->lengths[0];
		if(axes->elemType == connx_DataType_INT32) {
			for(uint32_t i = 0; i < axes_length; i++) {
				axes_array[i] = ((int32_t*)axes->base)[i];
			}
		} else if(axes->elemType == connx_DataType_INT64) {
			for(uint32_t i = 0; i < axes_length; i++) {
				axes_array[i] = ((int64_t*)axes->base)[i];
			}
		}

		// convert negative axes to positive
		for(uint32_t i = 0; i < axes_length; i++) {
			if(axes_array[i] < 0) {
				axes_array[i] += data->dimension;
			}
		}
	} else {
		axes_length = starts->lengths[0];
		for(uint32_t i = 0; i < axes_length; i++) {
			axes_array[i] = i;
		}
	}

	// normalize start, end and step
	connx_Tensor* old_starts = starts;
	connx_Tensor* old_ends = ends;
	connx_Tensor* old_steps = steps;

	uint32_t lengths[1] = { data->dimension };

	starts = connx_Tensor_create2(connx_DataType_INT32, 1, lengths);
	int32_t* starts_base = (int32_t*)starts->base;

	ends = connx_Tensor_create2(connx_DataType_INT32, 1, lengths);
	int32_t* ends_base = (int32_t*)ends->base;

	steps = connx_Tensor_create2(connx_DataType_INT32, 1, lengths);
	int32_t* steps_base = (int32_t*)steps->base;

	for(int32_t i = 0; i < (int32_t)data->dimension; i++) {
		for(uint32_t j = 0; j < axes_length; j++) {
			int32_t idx = axes_array[j];

			if(idx == i || (idx < 0 && idx + (int32_t)data->dimension - 1 == i)) {
				if(old_starts->elemType == connx_DataType_INT32) {
					starts_base[i] = ((int32_t*)old_starts->base)[j];
				} else {
					starts_base[i] = ((int64_t*)old_starts->base)[j];
				}

				if(old_ends->elemType == connx_DataType_INT32) {
					ends_base[i] = ((int32_t*)old_ends->base)[j];
				} else {
					ends_base[i] = ((int64_t*)old_ends->base)[j];
				}

				if(old_steps == NULL) {
					steps_base[i] = 1;
				} else if(old_steps->elemType == connx_DataType_INT32) {
					steps_base[i] = ((int32_t*)old_steps->base)[j];
				} else {
					steps_base[i] = ((int64_t*)old_steps->base)[j];
				}

				goto done;
			}
		}

		starts_base[i] = 0;
		ends_base[i] = data->lengths[i];
		steps_base[i] = 1;
done:
		;
	}

	connx_Stack_update(3, starts);
	connx_Stack_update(4, ends);
	connx_Stack_update(6, steps);

	if(axes != NULL) {
		connx_Stack_update(5, NULL);
	}

	// start/end boundary check
	for(uint32_t i = 0; i < data->dimension; i++) {
		// Convert negative index to positive
		if(starts_base[i] < 0)
			starts_base[i] += data->lengths[i];

		if(ends_base[i] < 0)
			ends_base[i] += data->lengths[i];

		// Convert boundary
		if(steps_base[i] > 0) {
			if(ends_base[i] > (int32_t)data->lengths[i]) {
				ends_base[i] = data->lengths[i];
			}

			if(starts_base[i] > ends_base[i]) {
				starts_base[i] = ends_base[i];
			}
		} else if(steps_base[i] < 0) {
			if(starts_base[i] >= (int32_t)data->lengths[i]) {
				starts_base[i] = data->lengths[i] - 1;
			}

			if(ends_base[i] >= starts_base[i]) {
				ends_base[i] = starts_base[i];
			}
		} else {
			connx_exception("steps[%" PRIu32 "] can not be zero", i);
			return false;
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

		output = connx_Tensor_create2(data->elemType, data->dimension, lengths);
		connx_Stack_update(1, output);
	}

	if(data->dimension != output->dimension) {
		connx_exception("data's dimension and output's dimension is differ");
		return false;
	}

	if(data->elemType != output->elemType) {
		connx_exception("data's type is differ from output's: %" PRIu32 ", expected: %" PRIu32, data->elemType, output->elemType);
		return false;
	}

	// Check output's lengths
	for(uint32_t i = 0; i < output->dimension; i++) {
		int32_t len = ends_base[i] - starts_base[i];
		int32_t rem = len % steps_base[i] == 0 ? 0 : 1;
		len /= steps_base[i];
		len += rem;

		if(len < 0) {
			connx_exception("steps[%" PRIu32 "]'s sign is incorrect: %" PRIu32, i, steps_base[i]);
			return false;
		}

		if((int32_t)output->lengths[i] != len) {
			connx_exception("output[%" PRIu32 "] length is incorrect: %" PRIu32 ", expected: %" PRIu32, i, output->lengths[i], len);
			return false;
		}
	}

	return true;
}

static bool Slice_exec(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];
	connx_Tensor* starts = (void*)stack[3];
	connx_Tensor* ends = (void*)stack[4];
	connx_Tensor* steps = (void*)stack[6];

	int32_t* starts_base = (int32_t*)starts->base;
	int32_t* ends_base = (int32_t*)ends->base;
	int32_t* steps_base = (int32_t*)steps->base;

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

	uint32_t elem_size = connx_DataType_size(output->elemType);
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

bool connx_opset_Slice_init() {
	connx_Operator_add("Slice", 1, 5, 0, Slice_resolve, Slice_exec,
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// output
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// data
		connx_DataType_TENSOR_INT32 | connx_DataType_TENSOR_INT64,	// starts
		connx_DataType_TENSOR_INT32 | connx_DataType_TENSOR_INT64,	// ends
		connx_DataType_TENSOR_INT32 | connx_DataType_TENSOR_INT64,	// axes
		connx_DataType_TENSOR_INT32 | connx_DataType_TENSOR_INT64);	// steps

	return true;
}
