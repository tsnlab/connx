#include <float.h>
#include <inttypes.h>
#include <string.h>
#include <connx/operator.h>
#include <connx/backend.h>

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

bool opset_ReduceMin(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* reduced = CONNX_GET_OUTPUT(0);
	connx_Tensor* data = CONNX_GET_INPUT(0);

	connx_AttributeInts* axes = CONNX_GET_ATTRIBUTE(0);
	__attribute__((unused)) connx_AttributeInt* keepdims = CONNX_GET_ATTRIBUTE(1);

	uint32_t dimension = data->dimension;

	// normalize axes
	uint32_t axes_length = axes->length;
	uint32_t axes_base[dimension];

	if(axes != NULL) {
		for(uint32_t i = 0; i < axes_length; i++) {
			axes_base[i] = axes->values[i] < 0 ? axes->values[i] + dimension : axes->values[i];
		}

		// sort axes_base
		for(uint32_t i = 0; i < axes_length - 1; i++) {
			for(uint32_t j = i; j < axes_length; j++) {
				if(axes_base[i] > axes_base[j]) {
					uint32_t tmp = axes_base[i];
					axes_base[i] = axes_base[j];
					axes_base[j] = tmp;
				}
			}
		}
	} else {
		for(uint32_t i = 0; i < data->dimension; i++) {
			axes_base[i] = i;
		}
	}

	uint32_t _units[data->dimension + 1];	// To make units[-1]
	uint32_t* units = _units + 1;
	units[data->dimension - 1] = 1;
	for(int32_t i = data->dimension - 1; i >= 0; i--) {
		int32_t idx = i - 1;
		units[idx] = units[idx + 1] * data->lengths[idx + 1];
	}

	uint32_t total = units[-1];
	uint32_t count = 1;

	switch(reduced->type) {
		case connx_UINT32: {
				uint32_t* output = NULL;
				uint32_t* input = NULL;

				for(uint32_t dim = 0, axes_idx = 0; dim < data->dimension; dim++) {
					if(axes_idx < axes_length && axes_base[axes_idx] == dim) {	// reduce
						total /= data->lengths[dim];

						if(input == NULL) {
							input = (uint32_t*)data->base;
						} else {
							if(input != (uint32_t*)data->base)
								backend->pal->free(backend->pal, input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (uint32_t*)reduced->base;
						} else {
							output = backend->pal->alloc(backend->pal, sizeof(uint32_t) * total);
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
					backend->pal->free(backend->pal, input);

				if(output != (uint32_t*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(uint32_t));
					backend->pal->free(backend->pal, output);
				}
			}
			break;
		case connx_INT32:
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
								backend->pal->free(backend->pal, input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (int32_t*)reduced->base;
						} else {
							output = backend->pal->alloc(backend->pal, sizeof(int32_t) * total);
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
					backend->pal->free(backend->pal, input);

				if(output != (int32_t*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(int32_t));
					backend->pal->free(backend->pal, output);
				}
			}
			break;
		case connx_UINT64:
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
								backend->pal->free(backend->pal, input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (uint64_t*)reduced->base;
						} else {
							output = backend->pal->alloc(backend->pal, sizeof(uint64_t) * total);
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
					backend->pal->free(backend->pal, input);

				if(output != (uint64_t*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(uint64_t));
					backend->pal->free(backend->pal, output);
				}
			}
			break;
		case connx_INT64:
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
								backend->pal->free(backend->pal, input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (int64_t*)reduced->base;
						} else {
							output = backend->pal->alloc(backend->pal, sizeof(int64_t) * total);
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
					backend->pal->free(backend->pal, input);

				if(output != (int64_t*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(int64_t));
					backend->pal->free(backend->pal, output);
				}
			}
			break;
		case connx_FLOAT32:
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
								backend->pal->free(backend->pal, input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (float*)reduced->base;
						} else {
							output = backend->pal->alloc(backend->pal, sizeof(float) * total);
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
					backend->pal->free(backend->pal, input);

				if(output != (float*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(float));
					backend->pal->free(backend->pal, output);
				}
			}
			break;
		case connx_FLOAT64:
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
								backend->pal->free(backend->pal, input);
							input = output;
						}

						if(dim + 1 >= data->dimension) {
							output = (double*)reduced->base;
						} else {
							output = backend->pal->alloc(backend->pal, sizeof(double) * total);
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
					backend->pal->free(backend->pal, input);

				if(output != (double*)reduced->base) {
					memcpy(reduced->base, output, connx_Tensor_total(reduced) * sizeof(double));
					backend->pal->free(backend->pal, output);
				}
			}
			break;
		default:
			backend->pal->error(backend->pal, "Illegal elemType: %" PRIu32, reduced->type);
			return false;
	}

	return true;
}
