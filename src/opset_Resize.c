#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <connx/operator.h>
#include <connx/backend.h>

#define DEBUG 0

#if DEBUG
#include <stdio.h>
#endif

// Ref: onnx/onnx/backend/test/case/node/resize.py
static float interpolate_1d_float32(uint32_t idx, float* data, uint32_t length, float scale, float* roi, char* coordinate_transformation_mode, float cubic_coeff_a, bool exclude_outside, float extrapolation_value, char* mode, char* nearest_mode) {
#if DEBUG
	printf("interpolate_1d: [%" PRIu32 "] ", idx);
	for(uint32_t i = 0; i < length; i++)
		printf("%f ", data[i]);
#endif

	// get original coordinate
	float origin_idx = 0;
	float output_length = scale * length;
	switch(coordinate_transformation_mode[3]) {
		case 'f':	// half_pixel
        	origin_idx = ((float)idx + 0.5) / scale - 0.5;
			break;
		case 'o':	// pytorch_half_pixel
			if(output_length == 1)
				origin_idx = -0.5;
			else
				origin_idx = (idx + 0.5) / scale - 0.5;
			break;
		case 'g':	// align_corners
			if(output_length == 1)
				origin_idx = 0;
			else
				origin_idx = (float)idx * (length - 1) / (output_length - 1);
			break;
		case 'm':	// asymmetric
			origin_idx = idx / scale;
			break;
		case 'h':	// tf_half_pixel_for_nn
			origin_idx = ((float)idx + 0.5) / scale;
			break;
		case 'c':	// tf_crop_and_resize
			if(output_length == 1)
				origin_idx = (roi[1] - roi[0]) * (length - 1) / 2;
			else
				origin_idx = idx * (roi[1] - roi[0]) * (length - 1) / (output_length - 1);

			origin_idx += roi[0] * (length - 1);

			if(origin_idx < 0 || origin_idx > length - 1)
				return extrapolation_value;
			break;
		default:
			abort();
	}

	int32_t origin_idx_int = origin_idx >= 0 ? (int32_t)origin_idx : (int32_t)origin_idx - 1;
	float ratio = 0;
	if(origin_idx == origin_idx_int)
		ratio = 1;
	else
		ratio = origin_idx - origin_idx_int;

	// get coeffects
	float coeffects[4];
	uint32_t coeffects_length = 0;
	switch(mode[0]) {
		case 'n':	// nearest
			if(ratio == (int32_t)ratio) {
				coeffects[0] = 0;
				coeffects[1] = 1;
				coeffects_length = 2;
			} else {
				switch(nearest_mode[0]) {
					case 'r':
						switch(nearest_mode[13]) {
							case 'f':	// round_prefer_floor
								if(ratio <= 0.5) {
									coeffects[0] = 1;
									coeffects[1] = 0;
								} else {
									coeffects[0] = 0;
									coeffects[1] = 1;
								}
								coeffects_length = 2;
								break;
							case 'c':	// round_prefer_ceil
								if(ratio < 0.5) {
									coeffects[0] = 1;
									coeffects[1] = 0;
								} else {
									coeffects[0] = 0;
									coeffects[1] = 1;
								}
								coeffects_length = 2;
								break;
							default:
								abort();
						}
						break;
					case 'f':	// floor
						coeffects[0] = 1;
						coeffects[1] = 0;
						coeffects_length = 2;
						break;
					case 'c':	// ceil
						coeffects[0] = 0;
						coeffects[1] = 1;
						coeffects_length = 2;
						break;
					default:
						abort();
				}
			}
			break;
		case 'l':	// linear
			coeffects[0] = 1 - ratio;
			coeffects[1] = ratio;
			coeffects_length = 2;
			break;
		case 'c':	// cubic
			coeffects[0] = ((cubic_coeff_a * (ratio + 1) - 5 * cubic_coeff_a) * (ratio + 1) + 8 * cubic_coeff_a) * (ratio + 1) - 4 * cubic_coeff_a;
            coeffects[1] = ((cubic_coeff_a + 2) * ratio - (cubic_coeff_a + 3)) * ratio * ratio + 1;
            coeffects[2] = ((cubic_coeff_a + 2) * (1 - ratio) - (cubic_coeff_a + 3)) * (1 - ratio) * (1 - ratio) + 1;
            coeffects[3] = ((cubic_coeff_a * ((1 - ratio) + 1) - 5 * cubic_coeff_a) * ((1 - ratio) + 1) + 8 * cubic_coeff_a) * ((1 - ratio) + 1) - 4 * cubic_coeff_a;
			coeffects_length = 4;
			break;
		default:
			abort();
	}

#if DEBUG
	printf("x=%f (int)x=%" PRId32 " ", origin_idx, origin_idx_int);
#endif
	// calculate base
	int32_t idx_base;
	if(origin_idx == origin_idx_int) {
		idx_base = origin_idx_int - coeffects_length / 2;
	} else {
		idx_base = origin_idx_int - coeffects_length / 2 + 1;
	}

	// exclude_outside
	if(exclude_outside) {
		float sum = 0;

		for(uint32_t i = 0; i < coeffects_length; i++) {
			int j = idx_base + i;
			if(j < 0 || (uint32_t)j >= length) {	// left or right edge padding
				coeffects[i] = 0;
			} else {
				sum += coeffects[i];
			}
		}

		if(sum != 0) {
			for(uint32_t i = 0; i < coeffects_length; i++) {
				coeffects[i] /= sum;
			}
		}
	}

#if DEBUG
	printf("coeffects: ");
	for(uint32_t i = 0; i < coeffects_length; i++) {
		printf("%f ", coeffects[i]);
	}

	printf("idxs=");
#endif
	float interpolate = 0;
	for(uint32_t i = 0; i < coeffects_length; i++) {
		float value;

		int j = idx_base + i;
#if DEBUG
		printf("%" PRId32 " ", j);
#endif
		if(j < 0) {	// left edge padding
			value = data[0];
		} else if((uint32_t)j >= length) {	// right edge padding
			value = data[length - 1];
		} else {
			value = data[j];
		}

		interpolate += coeffects[i] * value;
	}

#if DEBUG
	printf(", interpolate=%f\n", interpolate);
#endif
	return interpolate;
}

static float interpolate_nd_float32(uint32_t* idxs, float* data, uint32_t* lengths, float* scales, uint32_t dim, float* roi, char* coordinate_transformation_mode, float cubic_coeff_a, bool exclude_outside, float extrapolation_value, char* mode, char* nearest_mode) {
	if(dim == 1) {
		return interpolate_1d_float32(idxs[0], data, lengths[0], scales[0], roi, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
	} else {
		uint32_t unit = 1;
		for(uint32_t i = 0; i < dim - 1; i++)
			unit *= lengths[dim - i - 1];

		float interpolated[lengths[0]];
		for(uint32_t i = 0; i < lengths[0]; i++) {
			interpolated[i] = interpolate_nd_float32(idxs + 1, data + unit * i, lengths + 1, scales + 1,  dim - 1, roi + 2, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
		}

		return interpolate_1d_float32(idxs[0], interpolated, lengths[0], scales[0], roi, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
	}
}

bool opset_Resize(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	// outputs
	connx_Tensor* Y = CONNX_GET_OUTPUT(0);

	// inputs
	connx_Tensor* X = CONNX_GET_INPUT(0);
	connx_Tensor* roi = CONNX_GET_INPUT(1);
	connx_Tensor* scales = CONNX_GET_INPUT(2);
	connx_Tensor* sizes = CONNX_GET_INPUT(3);

	// attributes
	connx_AttributeString* coordinate_transformation_mode = CONNX_GET_ATTRIBUTE(0);
	connx_AttributeFloat* cubic_coeff_a = CONNX_GET_ATTRIBUTE(1);
	connx_AttributeInt* exclude_outside = CONNX_GET_ATTRIBUTE(2);
	connx_AttributeFloat* extrapolation_value = CONNX_GET_ATTRIBUTE(3);
	connx_AttributeString* mode = CONNX_GET_ATTRIBUTE(4);
	connx_AttributeString* nearest_mode = CONNX_GET_ATTRIBUTE(5);

	uint32_t dimension = X->dimension;

	// normalize sizes and scales
	int64_t sizes_base[dimension];
	float scales_base[dimension];

	// Create Y if NULL
	if(Y == NULL) {
		uint32_t lengths[dimension];

		if(sizes != NULL) {
			int64_t* sizes_base = (int64_t*)sizes->base;
			for(uint32_t i = 0; i < dimension; i++) {
				lengths[i] = sizes_base[i];
				scales_base[i] = (float)lengths[i] / (float)X->lengths[i];
			}
		} else {
			float* scales_base = (float*)scales->base;
			for(uint32_t i = 0; i < dimension; i++) {
				lengths[i] = sizes_base[i] = (int64_t)(scales_base[i] * X->lengths[i]);
			}
		}

		Y = connx_Tensor_create(backend->hal, X->type, dimension, lengths);
		CONNX_SET_OUTPUT(0, Y);
	}

	// Check all zero
	for(uint32_t i = 0; i < dimension; i++) {
		if(Y->lengths[i] != 0)
			goto not_all_zero;
	}

	return true;

not_all_zero:

	;
	float roi_base[dimension * 2];

	if(strncmp(coordinate_transformation_mode->value, "half_pixel", 11) == 0) {
	} else if(strncmp(coordinate_transformation_mode->value, "pytorch_half_pixel", 19) == 0) {
	} else if(strncmp(coordinate_transformation_mode->value, "align_corners", 14) == 0) {
	} else if(strncmp(coordinate_transformation_mode->value, "asymmetric", 11) == 0) {
	} else if(strncmp(coordinate_transformation_mode->value, "tf_half_pixel_for_nn", 21) == 0) {
	} else if(strncmp(coordinate_transformation_mode->value, "tf_crop_and_resize", 19) == 0) {
		// normalize roi
		float roi_normalized_base[dimension * 2];
		if(roi->type != connx_FLOAT32) {
			switch(roi->type) {
				case connx_FLOAT64:
					{
						double* roi_float64_base = (double*)roi->base;
						uint32_t total = connx_Tensor_total(roi);
						for(uint32_t i = 0; i < total; i++) {
							roi_normalized_base[i] = roi_float64_base[i];
						}
					}
					break;
				default:
					backend->hal->error(backend->hal, "Illegal roi type: %" PRIu32, roi->type);
					return false;
			}
		}

		// relocate (start1 ~ startN, end1 ~ endN) to (start1, end1 ~ startN, endN)
		for(uint32_t i = 0; i < dimension; i++) {
			roi_base[i * 2] = roi_normalized_base[i];
			roi_base[i * 2 + 1] = roi_normalized_base[i + dimension];
		}
	} else {
		backend->hal->error(backend->hal, "Illegal coordinate_transformation_mode: '%s'", coordinate_transformation_mode->value);
		return false;
	}

	// init idxs and lengths
	uint32_t lengths[dimension];
	uint32_t idxs[dimension];

	for(uint32_t i = 0; i < dimension; i++) {
		idxs[i] = 0;
		lengths[i] = sizes_base[i];
	}

	// non-fixed type variables
	switch(X->type) {
		case connx_FLOAT32: {
				float* Y_base = (float*)Y->base;
				float* X_base = (float*)X->base;

				while(true) {
					// process
#if DEBUG
					printf("\nidxs = ");
					for(uint32_t i = 0; i < dimension; i++)
						printf("%" PRIu32 " ", idxs[i]);
					printf("\n");
#endif

					*Y_base++ = interpolate_nd_float32(idxs, X_base, X->lengths, scales_base, X->dimension, roi_base, coordinate_transformation_mode->value, cubic_coeff_a->value, !!exclude_outside->value, extrapolation_value->value, mode->value, nearest_mode->value);

					// next
					for(int32_t dim = dimension - 1; dim >= 0; dim--) {
						if(++idxs[dim] >= lengths[dim]) {
							idxs[dim] = 0;
						} else {
							goto next_float32;
						}
					}

					break;

next_float32:
					;
				}
			}
			break;
		default:
			backend->hal->error(backend->hal, "Not supported type: %u\n", X->type);
			return false;
	}

	return true;
}
