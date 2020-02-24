#include <float.h>
#include <string.h>
#include <connx/connx.h>

static bool has_axis(uint32_t axis, int64_t* axes, uint32_t axes_length, uint32_t dimension) {
	for(uint32_t i = 0; i < axes_length; i++) {
		if((axes[i] < 0 && axes[i] + dimension == axis) || axes[i] == axis) {
			return true;
		}
	}

	return false;
}

static bool ReduceMin_resolve(uintptr_t* stack) {
	connx_Tensor* reduced = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];

	void* attr_axes = (void*)stack[3];
	int64_t* keepdims  = (void*)stack[4];

	// input and output's element type is same
	if(reduced->elemType != data->elemType) {
		connx_exception("reduced and data's element type is differ: %u != %u", data->elemType, reduced->elemType);
		return false;
	}

	// normalize axes
	if(attr_axes != NULL) {
		int64_t* axes_base = connx_Attribute_base(attr_axes);
		uint32_t axes_length = connx_Attribute_length(attr_axes);

		for(uint32_t i = 0; i < axes_length; i++) {
			if(axes_base[i] < 0) {
				axes_base[i] += data->dimension;
			}
		}

		for(uint32_t i = 0, axes_idx = 0; i < data->dimension; i++) {
			if(has_axis(i, axes_base, axes_length, data->dimension)) {
				for(uint32_t j = axes_idx; j < axes_length; j++) {
					if(axes_base[j] == i) {
						int64_t tmp = axes_base[axes_idx];
						axes_base[axes_idx] = i;
						axes_base[j] = tmp;
						break;
					}
				}
				axes_idx++;
			}
		}
	} else {
		int64_t array[data->dimension];
		for(uint32_t i = 0; i < data->dimension; i++) {
			array[i] = i;
		}

		stack[3] = connx_Attribute_create_ints(data->dimension, array);
		attr_axes = (void*)stack[3];
	}

	int64_t* axes_base = connx_Attribute_base(attr_axes);
	uint32_t axes_length = connx_Attribute_length(attr_axes);

	// Check reduced's lengths
	if(*keepdims != 0) {	// keepdims
		if(data->dimension != reduced->dimension) {
			connx_exception("reduced's dimension is differ from data's dimension: %u != %u", reduced->dimension, data->dimension);
			return false;
		}

		for(uint32_t i = 0, axes_idx = 0; i < data->dimension; i++) {
			if(axes_idx < axes_length && axes_base[axes_idx] == i) {
				axes_idx++;

				if(reduced->lengths[i] != 1) {
					connx_exception("reduced's length[%u] is not 1 but %u", i, reduced->lengths[i]);
					return false;
				}
			} else {
				if(reduced->lengths[i] != data->lengths[i]) {
					connx_exception("reduced's length[%u] is differ from data's length[%u]: %u != %u", i, i, reduced->lengths[i], data->lengths[i]);
					return false;
				}
			}
		}
	} else {				// do not keepdims
		if(data->dimension - axes_length == 0) {
			if(reduced->dimension != 1) {
				connx_exception("Illegal reduced's dimension: %u, expected: %u", reduced->dimension, 1);
				return false;
			}
		} else if(data->dimension - axes_length != reduced->dimension) {
			connx_exception("Illegal reduced's dimension: %u, expected: %u", reduced->dimension, data->dimension - axes_length);
			return false;
		}

		for(uint32_t i = 0, j = 0, axes_idx = 0; i < data->dimension; i++) {
			if(axes_idx < axes_length && axes_base[axes_idx] == i) {
				axes_idx++;
				continue;
			}

			if(reduced->lengths[j] != data->lengths[i]) {
				connx_exception("Illegal reduced's length[%u]: %u, expected: %u", j, reduced->lengths[j], data->lengths[i]);
				return false;
			}

			j++;
		}
	}

	return true;
}

static bool reduce_uint32(uint32_t* output, uint32_t* input, uint32_t length, uint32_t length2, uint32_t unit, uint32_t unit2, uint32_t count) {
	for(uint32_t i = 0; i < count; i++) {
		uint32_t* input2 = input;

		for(uint32_t j = 0; j < length2; j++) {
			uint32_t* input3 = input2;

			for(uint32_t k = 0; k < length; k++) {
				if(*output > *input3) {
					*output = *input3;
				}

				input3 += unit2;
			}

			input2++;
			output++;
		}

		input += unit;
	}

	return true;
}

static bool reduce_int32(int32_t* output, int32_t* input, uint32_t length, uint32_t length2, uint32_t unit, uint32_t unit2, uint32_t count) {
	for(uint32_t i = 0; i < count; i++) {
		int32_t* input2 = input;

		for(uint32_t j = 0; j < length2; j++) {
			int32_t* input3 = input2;

			for(uint32_t k = 0; k < length; k++) {
				if(*output > *input3) {
					*output = *input3;
				}

				input3 += unit2;
			}

			input2++;
			output++;
		}

		input += unit;
	}

	return true;
}

static bool reduce_uint64(uint64_t* output, uint64_t* input, uint32_t length, uint32_t length2, uint32_t unit, uint32_t unit2, uint32_t count) {
	for(uint32_t i = 0; i < count; i++) {
		uint64_t* input2 = input;

		for(uint32_t j = 0; j < length2; j++) {
			uint64_t* input3 = input2;

			for(uint32_t k = 0; k < length; k++) {
				if(*output > *input3) {
					*output = *input3;
				}

				input3 += unit2;
			}

			input2++;
			output++;
		}

		input += unit;
	}

	return true;
}

static bool reduce_int64(int64_t* output, int64_t* input, uint32_t length, uint32_t length2, uint32_t unit, uint32_t unit2, uint32_t count) {
	for(uint32_t i = 0; i < count; i++) {
		int64_t* input2 = input;

		for(uint32_t j = 0; j < length2; j++) {
			int64_t* input3 = input2;

			for(uint32_t k = 0; k < length; k++) {
				if(*output > *input3) {
					*output = *input3;
				}

				input3 += unit2;
			}

			input2++;
			output++;
		}

		input += unit;
	}

	return true;
}

static bool reduce_float32(float* output, float* input, uint32_t length, uint32_t length2, uint32_t unit, uint32_t unit2, uint32_t count) {
	for(uint32_t i = 0; i < count; i++) {
		float* input2 = input;

		for(uint32_t j = 0; j < length2; j++) {
			float* input3 = input2;

			for(uint32_t k = 0; k < length; k++) {
				if(*output > *input3) {
					*output = *input3;
				}

				input3 += unit2;
			}

			input2++;
			output++;
		}

		input += unit;
	}

	return true;
}

static bool reduce_float64(double* output, double* input, uint32_t length, uint32_t length2, uint32_t unit, uint32_t unit2, uint32_t count) {
	for(uint32_t i = 0; i < count; i++) {
		double* input2 = input;

		for(uint32_t j = 0; j < length2; j++) {
			double* input3 = input2;

			for(uint32_t k = 0; k < length; k++) {
				if(*output > *input3) {
					*output = *input3;
				}

				input3 += unit2;
			}

			input2++;
			output++;
		}

		input += unit;
	}

	return true;
}

static bool ReduceMin_exec(uintptr_t* stack) {
	connx_Tensor* reduced = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];

	void* attr_axes = (void*)stack[3];
	int64_t* axes_base = connx_Attribute_base(attr_axes);
	uint32_t axes_length = connx_Attribute_length(attr_axes);

	uint32_t _units[data->dimension + 1];	// To make units[-1]
	uint32_t* units = _units + 1;
	units[data->dimension - 1] = 1;
	for(int32_t i = data->dimension - 1; i >= 0; i--) {
		int32_t idx = i - 1;
		units[idx] = units[idx + 1] * data->lengths[idx + 1];
	}

	uint32_t total = units[-1];
	uint32_t count = 1;

	switch(reduced->elemType) {
		case connx_DataType_UINT32:
			{
				uint32_t* output = NULL;
				uint32_t* input = NULL;

				for(uint32_t dim = 0, axes_idx = 0; dim < data->dimension; dim++) {
					if(axes_idx < axes_length && axes_base[axes_idx] == dim) {	// reduce
						total /= data->lengths[dim];

						if(input == NULL) {
							input = (uint32_t*)data->base;
						} else {
							if(input != (uint32_t*)data->base)
								connx_free(input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (uint32_t*)reduced->base;
						} else {
							output = connx_alloc(sizeof(uint32_t) * total);
						}

						for(uint32_t i = 0; i < total; i++) {
							output[i] = UINT32_MAX;
						}

						reduce_uint32(output, input, 
								data->lengths[dim], dim + 1 < data->dimension ? data->lengths[dim + 1] : 1, 
								units[(int32_t)dim - 1], units[dim], 
								count);

						axes_idx++;
					} else {	// do not reduce
						count *= data->lengths[dim];
					}
				}

				if(input != (uint32_t*)data->base)
					connx_free(input);

				if(output != (uint32_t*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(uint32_t));
					connx_free(output);
				}
			}
			break;
		case connx_DataType_INT32:
			{
				int32_t* output = NULL;
				int32_t* input = NULL;

				for(uint32_t dim = 0, axes_idx = 0; dim < data->dimension; dim++) {
					if(axes_idx < axes_length && axes_base[axes_idx] == dim) {	// reduce
						total /= data->lengths[dim];

						if(input == NULL) {
							input = (int32_t*)data->base;
						} else {
							if(input != (int32_t*)data->base)
								connx_free(input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (int32_t*)reduced->base;
						} else {
							output = connx_alloc(sizeof(int32_t) * total);
						}

						for(uint32_t i = 0; i < total; i++) {
							output[i] = INT32_MAX;
						}

						reduce_int32(output, input, 
								data->lengths[dim], dim + 1 < data->dimension ? data->lengths[dim + 1] : 1, 
								units[(int32_t)dim - 1], units[dim], 
								count);

						axes_idx++;
					} else {	// do not reduce
						count *= data->lengths[dim];
					}
				}

				if(input != (int32_t*)data->base)
					connx_free(input);

				if(output != (int32_t*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(int32_t));
					connx_free(output);
				}
			}
			break;
		case connx_DataType_UINT64:
			{
				uint64_t* output = NULL;
				uint64_t* input = NULL;

				for(uint32_t dim = 0, axes_idx = 0; dim < data->dimension; dim++) {
					if(axes_idx < axes_length && axes_base[axes_idx] == dim) {	// reduce
						total /= data->lengths[dim];

						if(input == NULL) {
							input = (uint64_t*)data->base;
						} else {
							if(input != (uint64_t*)data->base)
								connx_free(input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (uint64_t*)reduced->base;
						} else {
							output = connx_alloc(sizeof(uint64_t) * total);
						}

						for(uint32_t i = 0; i < total; i++) {
							output[i] = UINT64_MAX;
						}

						reduce_uint64(output, input, 
								data->lengths[dim], dim + 1 < data->dimension ? data->lengths[dim + 1] : 1, 
								units[(int32_t)dim - 1], units[dim], 
								count);

						axes_idx++;
					} else {	// do not reduce
						count *= data->lengths[dim];
					}
				}

				if(input != (uint64_t*)data->base)
					connx_free(input);

				if(output != (uint64_t*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(uint64_t));
					connx_free(output);
				}
			}
			break;
		case connx_DataType_INT64:
			{
				int64_t* output = NULL;
				int64_t* input = NULL;

				for(uint32_t dim = 0, axes_idx = 0; dim < data->dimension; dim++) {
					if(axes_idx < axes_length && axes_base[axes_idx] == dim) {	// reduce
						total /= data->lengths[dim];

						if(input == NULL) {
							input = (int64_t*)data->base;
						} else {
							if(input != (int64_t*)data->base)
								connx_free(input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (int64_t*)reduced->base;
						} else {
							output = connx_alloc(sizeof(int64_t) * total);
						}

						for(uint32_t i = 0; i < total; i++) {
							output[i] = INT64_MAX;
						}

						reduce_int64(output, input, 
								data->lengths[dim], dim + 1 < data->dimension ? data->lengths[dim + 1] : 1, 
								units[(int32_t)dim - 1], units[dim], 
								count);

						axes_idx++;
					} else {	// do not reduce
						count *= data->lengths[dim];
					}
				}

				if(input != (int64_t*)data->base)
					connx_free(input);

				if(output != (int64_t*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(int64_t));
					connx_free(output);
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* output = NULL;
				float* input = NULL;

				for(uint32_t dim = 0, axes_idx = 0; dim < data->dimension; dim++) {
					if(axes_idx < axes_length && axes_base[axes_idx] == dim) {	// reduce
						total /= data->lengths[dim];

						if(input == NULL) {
							input = (float*)data->base;
						} else {
							if(input != (float*)data->base)
								connx_free(input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (float*)reduced->base;
						} else {
							output = connx_alloc(sizeof(float) * total);
						}

						for(uint32_t i = 0; i < total; i++) {
							output[i] = FLT_MAX;
						}

						reduce_float32(output, input, 
								data->lengths[dim], dim + 1 < data->dimension ? data->lengths[dim + 1] : 1, 
								units[(int32_t)dim - 1], units[dim], 
								count);

						axes_idx++;
					} else {	// do not reduce
						count *= data->lengths[dim];
					}
				}

				if(input != (float*)data->base)
					connx_free(input);

				if(output != (float*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(float));
					connx_free(output);
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* output = NULL;
				double* input = NULL;

				for(uint32_t dim = 0, axes_idx = 0; dim < data->dimension; dim++) {
					if(axes_idx < axes_length && axes_base[axes_idx] == dim) {	// reduce
						total /= data->lengths[dim];

						if(input == NULL) {
							input = (double*)data->base;
						} else {
							if(input != (double*)data->base)
								connx_free(input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (double*)reduced->base;
						} else {
							output = connx_alloc(sizeof(double) * total);
						}

						for(uint32_t i = 0; i < total; i++) {
							output[i] = DBL_MAX;
						}

						reduce_float64(output, input, 
								data->lengths[dim], dim + 1 < data->dimension ? data->lengths[dim + 1] : 1, 
								units[(int32_t)dim - 1], units[dim], 
								count);

						axes_idx++;
					} else {	// do not reduce
						count *= data->lengths[dim];
					}
				}

				if(input != (double*)data->base)
					connx_free(input);

				if(output != (double*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(double));
					connx_free(output);
				}
			}
			break;
		default:
			connx_exception("Illegal elemType: %u", reduced->elemType);
			return false;
	}

	return true;
}

bool connx_opset_ReduceMin_init() {
	uint32_t type = connx_DataType_TENSOR |
		connx_DataType_UINT32 | connx_DataType_INT32 | 
		connx_DataType_UINT64 | connx_DataType_INT64 | 
		connx_DataType_FLOAT;

	connx_Operator_add("ReduceMin", 1, 1, 2, ReduceMin_resolve, ReduceMin_exec,
		type,	// reduced
		type,	// data
		"axes", connx_DataType_INT64_ARRAY, 0, NULL,
		"keepdims", connx_DataType_INT64, 1);

	return true;
}
