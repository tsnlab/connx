#include <stdlib.h>
#include <string.h>
#include <connx/connx.h>

#define DEBUG 0
static bool Resize_resolve(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	connx_Tensor* roi = (void*)stack[3];
	connx_Tensor* scales = (void*)stack[4];
	connx_Tensor* sizes = (void*)stack[5];	// optional
	char* coordinate_transformation_mode = (void*)stack[6];
	__attribute__((unused)) float* cubic_coeff_a = (void*)stack[7];
	__attribute__((unused)) int64_t* exclude_outside = (void*)stack[8];
	__attribute__((unused)) float* extrapolation_value = (void*)stack[9];
	char* mode = (void*)stack[10];
	char* nearest_mode = (void*)stack[11];

	// Create Y if NULL
	if(Y == NULL) {
		uint32_t lengths[X->dimension];

		if(sizes != NULL) {
			int64_t* sizes_base = (int64_t*)sizes->base;
			for(uint32_t i = 0; i < X->dimension; i++) {
				lengths[i] = sizes_base[i];
			}
		} else {
			float* scales_base = (float*)scales->base;
			for(uint32_t i = 0; i < X->dimension; i++) {
				lengths[i] = (int64_t)(scales_base[i] * X->lengths[i]);
			}
		}

		Y = connx_Tensor_create2(X->elemType, X->dimension, lengths);
		connx_Operator_stack_update(Y, 1, 1);
	}

	// Check X and Y's dimension
	if(X->dimension != Y->dimension) {
		connx_exception("X's dimension and Y's dimension are different %u != %u", X->dimension, Y->dimension);
		return false;
	}

	// Check X and Y's type
	if(X->type != Y->type) {
		connx_exception("X and Y's type is differ: X: %u, Y: %u", X->type, Y->type);
		return false;
	}

	// Set scales or sizes
	if(sizes != NULL) {	// make scales from size
		if(sizes->dimension != 1) {
			connx_exception("sizes' dimension is not 1 but %u", sizes->dimension);
			return false;
		}
		
		if(sizes->lengths[0] != X->dimension) {
			connx_exception("sizes' length is different from X and Y's: %u != %u", sizes->lengths[0], X->dimension);
			return false;
		}

		int64_t* sizes_base = (int64_t*)sizes->base;
		for(uint32_t i = 0; i < Y->dimension; i++) {
			if(Y->lengths[i] != sizes_base[i]) {
				connx_exception("output size is mismatch: Y[%u]'s length is %u but sizes[%u] is %u", i, Y->lengths[i], i, sizes_base[i]);
				return false;
			}
		}

		connx_Tensor_delete(scales);
		scales = connx_Tensor_create(connx_DataType_FLOAT32, 1, X->dimension);
		stack[4] = (uintptr_t)(void*)scales;

		float* scales_base = (float*)scales->base;
		for(uint32_t i = 0; i < X->dimension; i++) {
			scales_base[i] = (float)Y->lengths[i] / (float)X->lengths[i];
		}
	} else {			// make sizes from scales
		if(scales->dimension != 1) {
			connx_exception("scales' dimension is not 1 but %u", scales->dimension);
			return false;
		}
		
		if(scales->lengths[0] != X->dimension) {
			connx_exception("scales' length is different from X and Y's: %u != %u", scales->lengths[0], X->dimension);
			return false;
		}

		sizes = connx_Tensor_create(connx_DataType_INT64, 1, X->dimension);
		stack[5] = (uintptr_t)(void*)sizes;

		float* scales_base = (float*)scales->base;
		int64_t* sizes_base = (int64_t*)sizes->base;
		for(uint32_t i = 0; i < X->dimension; i++) {
			sizes_base[i] = (int64_t)(scales_base[i] * X->lengths[i]);
		}

		for(uint32_t i = 0; i < Y->dimension; i++) {
			if(sizes_base[i] != Y->lengths[i]) {
				connx_exception("scaled output mismatches: Y[%u]'s length is %u, expected: %ld", i, Y->lengths[i], sizes_base[i]);
				return false;
			}
		}
	}

	if(strncmp(coordinate_transformation_mode, "half_pixel", 11) == 0) {
	} else if(strncmp(coordinate_transformation_mode, "pytorch_half_pixel", 19) == 0) {
	} else if(strncmp(coordinate_transformation_mode, "align_corners", 14) == 0) {
	} else if(strncmp(coordinate_transformation_mode, "asymmetric", 11) == 0) {
	} else if(strncmp(coordinate_transformation_mode, "tf_half_pixel_for_nn", 21) == 0) {
	} else if(strncmp(coordinate_transformation_mode, "tf_crop_and_resize", 19) == 0) {
		if(roi->dimension != 1) {
			connx_exception("roi must be 1 dimension but %u", roi->dimension);
			return false;
		}

		if(roi->elemType != connx_DataType_FLOAT32) {
			connx_Tensor* new_roi = connx_Tensor_create(connx_DataType_FLOAT32, roi->dimension, roi->lengths);
			stack[3] = (uintptr_t)(void*)new_roi;

			float* new_roi_base = (float*)new_roi->base;

			switch(roi->type) {
				case connx_DataType_FLOAT64:
					{
						double* roi_base = (double*)roi->base;
						uint32_t total = connx_Tensor_total(roi);
						for(uint32_t i = 0; i < total; i++) {
							*new_roi_base++ = *roi_base++;
						}
					}
					break;
				default:
					connx_exception("Illegal roi type: %u", roi->type);
					return false;
			}

			connx_Tensor_delete(roi);
			roi = new_roi;
		}

		// relocate (start1 ~ startN, end1 ~ endN) to (start1, end1 ~ startN, endN)
		float* roi_base = (float*)roi->base;
		float roi2[roi->lengths[0]];
		memcpy(roi2, roi_base, sizeof(float) * roi->lengths[0]);

		uint32_t len = roi->lengths[0] / 2;
		for(uint32_t i = 0; i < len; i++) {
			roi_base[i * 2] = roi2[i];
			roi_base[i * 2 + 1] = roi2[i + len];
		}
	} else {
		connx_exception("Illegal coordinate_transformation_mode: '%s'", coordinate_transformation_mode);
		return false;
	}

	// coeffects
	if(strncmp(mode, "nearest", 8) == 0) {
		if(strncmp(nearest_mode, "round_prefer_floor", 19) == 0) {
		} else if(strncmp(nearest_mode, "round_prefer_ceil", 18) == 0) {
		} else if(strncmp(nearest_mode, "floor", 6) == 0) {
		} else if(strncmp(nearest_mode, "ceil", 5) == 0) {
		} else {
			connx_exception("Illegal nearest_mode: '%s'", nearest_mode);
			return false;
		}
	} else if(strncmp(mode, "linear", 7) == 0) {
	} else if(strncmp(mode, "cubic", 6) == 0) {
	} else {
		connx_exception("Illegal mode: '%s'", mode);
		return false;
	}

	return true;
}

// Ref: onnx/onnx/backend/test/case/node/resize.py
static float interpolate_1d_float32(uint32_t idx, float* data, uint32_t length, float scale, float* roi, char* coordinate_transformation_mode, float cubic_coeff_a, bool exclude_outside, float extrapolation_value, char* mode, char* nearest_mode) {
#if DEBUG
	printf("interpolate_1d: [%u] ", idx);
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
	printf("x=%f (int)x=%d ", origin_idx, origin_idx_int);
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
		printf("%d ", j);
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

static bool Resize_exec(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	connx_Tensor* roi = (void*)stack[3];
	connx_Tensor* scales = (void*)stack[4];
	connx_Tensor* sizes = (void*)stack[5];	// optional
	char* coordinate_transformation_mode = (void*)stack[6];
	float* cubic_coeff_a = (void*)stack[7];
	int64_t* exclude_outside = (void*)stack[8];
	float* extrapolation_value = (void*)stack[9];
	char* mode = (void*)stack[10];
	char* nearest_mode = (void*)stack[11];

	uint32_t dimension = Y->dimension;
	uint32_t lengths[dimension];
	uint32_t idxs[dimension];

	bool has_next = false;
	int64_t* sizes_base = (int64_t*)sizes->base;
	for(uint32_t i = 0; i < dimension; i++) {
		idxs[i] = 0;
		lengths[i] = sizes_base[i];

		if(lengths[i] > 0)
			has_next = true;
	}

	if(!has_next)
		return true;

	// fixed type variables
	float* scales_base = (float*)scales->base;
	float* roi_base = (float*)roi->base;

	// non-fixed type variables
	float* Y_base = (float*)Y->base;
	float* X_base = (float*)X->base;

	while(true) {
		// process
#if DEBUG
		printf("\nidxs = ");
		for(uint32_t i = 0; i < dimension; i++)
			printf("%u ", idxs[i]);
		printf("\n");
#endif

		*Y_base++ = interpolate_nd_float32(idxs, X_base, X->lengths, scales_base, X->dimension, roi_base, coordinate_transformation_mode, *cubic_coeff_a, !!*exclude_outside, *extrapolation_value, mode, nearest_mode);

		// next
		for(int32_t dim = dimension - 1; dim >= 0; dim--) {
			if(++idxs[dim] >= lengths[dim]) {
				idxs[dim] = 0;
			} else {
				goto next;
			}
		}

		break;

next:
		;
	}

	return true;
}

bool connx_opset_Resize_init() {
	connx_Operator_add("Resize", 1, 4, 6, Resize_resolve, Resize_exec,
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// Y
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// X
		connx_DataType_TENSOR_FLOAT,	// roi
		connx_DataType_TENSOR_FLOAT32,	// scales
		connx_DataType_TENSOR_INT64,	// sizes (optional)
		"coordinate_transformation_mode", connx_DataType_STRING, "half_pixel",
		"cubic_coeff_a", connx_DataType_FLOAT32, -0.75,
		"exclude_outside", connx_DataType_INT64, 0,
		"extrapolation_value", connx_DataType_FLOAT32, 0.0,
		"mode", connx_DataType_STRING, "nearest",
		"nearest_mode", connx_DataType_STRING, "round_prefer_floor");

	return true;
}
