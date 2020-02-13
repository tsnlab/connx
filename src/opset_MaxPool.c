#include <stdlib.h>
#include <float.h>
#include <connx/connx.h>

static bool MaxPool_resolve(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	void* attr_auto_pad = (void*)stack[3];
	void* attr_ceil_mode = (void*)stack[4];
	void* attr_dilations = (void*)stack[5];
	void* attr_kernel_shape = (void*)stack[6];
	void* attr_pads = (void*)stack[7];
	__attribute__((unused)) void* attr_storage_order = (void*)stack[8];
	void* attr_strides = (void*)stack[9];

	char* auto_pad = (char*)attr_auto_pad;
	uint32_t kernel_shape_length = connx_Attribute_length(attr_kernel_shape);
	int64_t* kernel_shape = connx_Attribute_base(attr_kernel_shape);
	int64_t* ceil_mode = (int64_t*)attr_ceil_mode;
	uint32_t dilations_length = connx_Attribute_length(attr_dilations);
	int64_t* dilations = connx_Attribute_base(attr_dilations);
	uint32_t strides_length = connx_Attribute_length(attr_strides);
	int64_t* strides = connx_Attribute_base(attr_strides);
	uint32_t pads_length = connx_Attribute_length(attr_pads);
	int64_t* pads = connx_Attribute_base(attr_pads);

	if(dilations_length == 0) {
		int64_t array[kernel_shape_length];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			array[i] = 1;
		}

		connx_Attribute_delete(attr_dilations);
		stack[5] = connx_Attribute_create_ints(kernel_shape_length, array);
		attr_dilations = (void*)stack[5];
		dilations_length = connx_Attribute_length(attr_dilations);
		dilations = connx_Attribute_base(attr_dilations);
	}

	if(auto_pad[0] == 'S') {
		int64_t array[kernel_shape_length * 2];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			int64_t pad = (Y->lengths[i] - 1) * strides[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - X->lengths[i];
			if(pad % 2 == 1) {
				array[i] = array[i + kernel_shape_length] = pad / 2;
				if(auto_pad[5] == 'U') {	// SAME_UPPER
					array[i + kernel_shape_length]++;
				} else {					// SAME_LOWER
					array[i]++;
				}
			} else {
				array[i] = array[i + kernel_shape_length] = pad / 2;
			}
		}

		connx_Attribute_delete(attr_pads);
		stack[7] = connx_Attribute_create_ints(kernel_shape_length * 2, array);
		attr_pads = (void*)stack[7];
		pads_length = connx_Attribute_length(attr_pads);
		pads = connx_Attribute_base(attr_pads);
	} else if(auto_pad[0] == 'V' || pads_length == 0) {
		int64_t array[kernel_shape_length * 2];
		for(uint32_t i = 0; i < kernel_shape_length * 2; i++) {
			array[i] = 0;
		}

		connx_Attribute_delete(attr_pads);
		stack[7] = connx_Attribute_create_ints(kernel_shape_length * 2, array);
		attr_pads = (void*)stack[7];
		pads_length = connx_Attribute_length(attr_pads);
		pads = connx_Attribute_base(attr_pads);
	}

	if(strides_length == 0) {
		int64_t array[kernel_shape_length];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			array[i] = 1;
		}

		connx_Attribute_delete(attr_strides);
		stack[9] = connx_Attribute_create_ints(kernel_shape_length, array);
		attr_strides = (void*)stack[9];
		strides = connx_Attribute_base(attr_strides);
		strides_length = connx_Attribute_length(attr_strides);
	}

	if(X->dimension != Y->dimension) {
		connx_exception("X's dimension and Y's dimension is not same: X: %u, Y: %u", X->dimension, Y->dimension);
		return false;
	}

	for(uint32_t i = 0; i < X->dimension - kernel_shape_length; i++) {
		if(X->lengths[i] != Y->lengths[i]) {
			connx_exception("X's shape and Y's shape is not same: X[%u]: %u, Y[%u]: %u", i, X->lengths[i], i, Y->lengths[i]);
			return false;
		}
	}

	for(uint32_t i = 0; i < kernel_shape_length; i++) {
		uint32_t v1 = (X->lengths[X->dimension - kernel_shape_length + i] + pads[i] + pads[i + kernel_shape_length] - ((kernel_shape[i] - 1) * dilations[i] + 1));
		uint32_t length = (uint32_t)(v1 / strides[i] + 1);
		if(*ceil_mode != 0) {
			if(v1 % strides[i] != 0) {
				length++;
				pads[i + kernel_shape_length]++;
			}
		}

		if(Y->lengths[Y->dimension - kernel_shape_length + i] != length) {
			connx_exception("Illegal shape, Y's length[%u] is %u, expected: %u", Y->dimension - kernel_shape_length + i, Y->lengths[Y->dimension - kernel_shape_length + i], length);
			return false;
		}
	}

	if(X->dimension < kernel_shape_length) {
		connx_exception("X's dimension: %u is smaller than kernel_shape's dimension: %u", X->dimension, kernel_shape_length);
		return false;
	}

	if(kernel_shape_length != 1 && kernel_shape_length != 2 && kernel_shape_length != 3) {
		connx_exception("kernel_shape count must be 1, 2 or 3 but %u", kernel_shape_length);
		return false;
	}

	if(kernel_shape_length != dilations_length) {
		connx_exception("dilation shape dimension: %u is different to kernel_shape's dimension: %u", dilations_length, kernel_shape_length);
		return false;
	}

	if(kernel_shape_length != strides_length) {
		connx_exception("stride shape dimension: %u is different to kernel_shape's dimension: %u", strides_length, kernel_shape_length);
		return false;
	}

	if(kernel_shape_length * 2 != pads_length) {
		connx_exception("pads shape dimension: %u is different to kernel_shape's dimension: %u", pads_length, kernel_shape_length);
		return false;
	}

	return true;
}

static void* pool_1d_uint32(uint32_t* Y, uint32_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	uint32_t tmp, tmp2;

	for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = 0;
		for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int64_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_uint32(uint32_t* Y, uint32_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	uint32_t tmp, tmp2;

	for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = 0;
			for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int64_t t1 = d1 + kd1 + dilations[0] - 1;
					int64_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1]) {
						tmp2 = X[t2 * X_lengths[1] + t1];
						if(tmp2 > tmp)
							tmp = tmp2;
					}
				}
			}
			*Y++ = tmp;
		}
	}

	return Y;
}

static void* pool_3d_uint32(uint32_t* Y, uint32_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	uint32_t tmp, tmp2;

	for(int64_t d3 = -pads[2]; d3 <= X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = 0;
				for(int64_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int64_t t1 = d1 + kd1 + dilations[0] - 1;
							int64_t t2 = d2 + kd2 + dilations[1] - 1;
							int64_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1] && t3 >= 0 && t3 < X_lengths[2]) {
								tmp2 = X[t3 * X_lengths[1] * X_lengths[2] + t2 * X_lengths[2] + t1];
								if(tmp2 > tmp)
									tmp = tmp2;
							}
						}
					}
				}
				*Y++ = tmp;
			}
		}
	}

	return Y;
}

static void* pool_1d_uint64(uint64_t* Y, uint64_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	uint64_t tmp, tmp2;

	for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = 0;
		for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int64_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_uint64(uint64_t* Y, uint64_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	uint64_t tmp, tmp2;

	for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = 0;
			for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int64_t t1 = d1 + kd1 + dilations[0] - 1;
					int64_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1]) {
						tmp2 = X[t2 * X_lengths[1] + t1];
						if(tmp2 > tmp)
							tmp = tmp2;
					}
				}
			}
			*Y++ = tmp;
		}
	}

	return Y;
}

static void* pool_3d_uint64(uint64_t* Y, uint64_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	uint64_t tmp, tmp2;

	for(int64_t d3 = -pads[2]; d3 <= X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = 0;
				for(int64_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int64_t t1 = d1 + kd1 + dilations[0] - 1;
							int64_t t2 = d2 + kd2 + dilations[1] - 1;
							int64_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1] && t3 >= 0 && t3 < X_lengths[2]) {
								tmp2 = X[t3 * X_lengths[1] * X_lengths[2] + t2 * X_lengths[2] + t1];
								if(tmp2 > tmp)
									tmp = tmp2;
							}
						}
					}
				}
				*Y++ = tmp;
			}
		}
	}

	return Y;
}

static void* pool_1d_int32(int32_t* Y, int32_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	int32_t tmp, tmp2;

	for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = INT32_MIN;
		for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int64_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_int32(int32_t* Y, int32_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	int32_t tmp, tmp2;

	for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = INT32_MIN;
			for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int64_t t1 = d1 + kd1 + dilations[0] - 1;
					int64_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1]) {
						tmp2 = X[t2 * X_lengths[1] + t1];
						if(tmp2 > tmp)
							tmp = tmp2;
					}
				}
			}
			*Y++ = tmp;
		}
	}

	return Y;
}

static void* pool_3d_int32(int32_t* Y, int32_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	int32_t tmp, tmp2;

	for(int64_t d3 = -pads[2]; d3 <= X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = INT32_MIN;
				for(int64_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int64_t t1 = d1 + kd1 + dilations[0] - 1;
							int64_t t2 = d2 + kd2 + dilations[1] - 1;
							int64_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1] && t3 >= 0 && t3 < X_lengths[2]) {
								tmp2 = X[t3 * X_lengths[1] * X_lengths[2] + t2 * X_lengths[2] + t1];
								if(tmp2 > tmp)
									tmp = tmp2;
							}
						}
					}
				}
				*Y++ = tmp;
			}
		}
	}

	return Y;
}

static void* pool_1d_int64(int64_t* Y, int64_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	int64_t tmp, tmp2;

	for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = INT64_MIN;
		for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int64_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_int64(int64_t* Y, int64_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	int64_t tmp, tmp2;

	for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = INT64_MIN;
			for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int64_t t1 = d1 + kd1 + dilations[0] - 1;
					int64_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1]) {
						tmp2 = X[t2 * X_lengths[1] + t1];
						if(tmp2 > tmp)
							tmp = tmp2;
					}
				}
			}
			*Y++ = tmp;
		}
	}

	return Y;
}

static void* pool_3d_int64(int64_t* Y, int64_t* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	int64_t tmp, tmp2;

	for(int64_t d3 = -pads[2]; d3 <= X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = INT64_MIN;
				for(int64_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int64_t t1 = d1 + kd1 + dilations[0] - 1;
							int64_t t2 = d2 + kd2 + dilations[1] - 1;
							int64_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1] && t3 >= 0 && t3 < X_lengths[2]) {
								tmp2 = X[t3 * X_lengths[1] * X_lengths[2] + t2 * X_lengths[2] + t1];
								if(tmp2 > tmp)
									tmp = tmp2;
							}
						}
					}
				}
				*Y++ = tmp;
			}
		}
	}

	return Y;
}

static void* pool_1d_float32(float* Y, float* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	float tmp, tmp2;

	for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = -FLT_MAX;
		for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int64_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_float32(float* Y, float* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	float tmp, tmp2;

	for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = -FLT_MAX;
			for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int64_t t1 = d1 + kd1 + dilations[0] - 1;
					int64_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1]) {
						tmp2 = X[t2 * X_lengths[1] + t1];
						if(tmp2 > tmp)
							tmp = tmp2;
					}
				}
			}
			*Y++ = tmp;
		}
	}

	return Y;
}

static void* pool_3d_float32(float* Y, float* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	float tmp, tmp2;

	for(int64_t d3 = -pads[2]; d3 <= X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = -FLT_MAX;
				for(int64_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int64_t t1 = d1 + kd1 + dilations[0] - 1;
							int64_t t2 = d2 + kd2 + dilations[1] - 1;
							int64_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1] && t3 >= 0 && t3 < X_lengths[2]) {
								tmp2 = X[t3 * X_lengths[1] * X_lengths[2] + t2 * X_lengths[2] + t1];
								if(tmp2 > tmp)
									tmp = tmp2;
							}
						}
					}
				}
				*Y++ = tmp;
			}
		}
	}

	return Y;
}

static void* pool_1d_float64(double* Y, double* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	double tmp, tmp2;

	for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = -FLT_MAX;
		for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int64_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_float64(double* Y, double* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	float tmp, tmp2;

	for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = -FLT_MAX;
			for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int64_t t1 = d1 + kd1 + dilations[0] - 1;
					int64_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1]) {
						tmp2 = X[t2 * X_lengths[1] + t1];
						if(tmp2 > tmp)
							tmp = tmp2;
					}
				}
			}
			*Y++ = tmp;
		}
	}

	return Y;
}

static void* pool_3d_float64(double* Y, double* X, uint32_t* X_lengths, int64_t* kernel, int64_t* pads, int64_t* strides, int64_t* dilations) {
	double tmp, tmp2;

	for(int64_t d3 = -pads[2]; d3 <= X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int64_t d2 = -pads[1]; d2 <= X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int64_t d1 = -pads[0]; d1 <= X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = -DBL_MAX;
				for(int64_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int64_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int64_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int64_t t1 = d1 + kd1 + dilations[0] - 1;
							int64_t t2 = d2 + kd2 + dilations[1] - 1;
							int64_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < X_lengths[0] && t2 >= 0 && t2 < X_lengths[1] && t3 >= 0 && t3 < X_lengths[2]) {
								tmp2 = X[t3 * X_lengths[1] * X_lengths[2] + t2 * X_lengths[2] + t1];
								if(tmp2 > tmp)
									tmp = tmp2;
							}
						}
					}
				}
				*Y++ = tmp;
			}
		}
	}

	return Y;
}

// Ref: onnx/backend/test/case/node/pool_op_common.py
static bool MaxPool_exec(uintptr_t* stack) {
	// TODO: implement ceil_mode and t dilations_count
	// TODO: ref: https://datascience.stackexchange.com/questions/28881/what-is-dilated-pooling-and-how-it-works-mathematically

	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	__attribute__((unused)) char* attr_auto_pad = (void*)stack[3];
	__attribute__((unused)) void* attr_ceil_mode = (void*)stack[4];
	void* attr_dilations = (void*)stack[5];
	void* attr_kernel_shape = (void*)stack[6];
	void* attr_pads = (void*)stack[7];
	__attribute__((unused)) void* attr_storage_order = (void*)stack[8];
	void* attr_strides = (void*)stack[9];

	int64_t* kernel_shape = connx_Attribute_base(attr_kernel_shape);
	uint32_t kernel_shape_length = connx_Attribute_length(attr_kernel_shape);
	__attribute__((unused)) int64_t* dilations = connx_Attribute_base(attr_dilations);
	__attribute__((unused)) uint32_t dilations_length = connx_Attribute_length(attr_dilations);
	int64_t* strides = connx_Attribute_base(attr_strides);
	__attribute__((unused)) uint32_t strides_length = connx_Attribute_length(attr_strides);
	int64_t* pads = connx_Attribute_base(attr_pads);
	__attribute__((unused)) uint32_t pads_length = connx_Attribute_length(attr_pads);

	uint32_t* X_lengths = X->lengths + X->dimension - kernel_shape_length;
	uint32_t unit = 1;
	uint32_t loop = 1;
	for(uint32_t i = 0; i < X->dimension; i++) {
		if(i < X->dimension - kernel_shape_length) {
			loop *= X->lengths[i];
		} else {
			unit *= X->lengths[i];
		}
	}

	unit *= connx_DataType_size(X->elemType);

	void* X_base = X->base;
	void* Y_base = Y->base;

	switch(X->elemType) {
		case connx_DataType_UINT32:
			if(kernel_shape_length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_uint32(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_uint32(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_uint32(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else {
				connx_exception("Kernel shape not supported: %u\n", kernel_shape_length);
				return false;
			}
			break;
		case connx_DataType_UINT64:
			if(kernel_shape_length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_uint64(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_uint64(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_uint64(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else {
				connx_exception("Kernel shape not supported: %u\n", kernel_shape_length);
				return false;
			}
			break;
		case connx_DataType_INT32:
			if(kernel_shape_length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_int32(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_int32(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_int32(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else {
				connx_exception("Kernel shape not supported: %u\n", kernel_shape_length);
				return false;
			}
			break;
		case connx_DataType_INT64:
			if(kernel_shape_length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_int64(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_int64(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_int64(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else {
				connx_exception("Kernel shape not supported: %u\n", kernel_shape_length);
				return false;
			}
			break;
		case connx_DataType_FLOAT16:
		case connx_DataType_FLOAT32:
			if(kernel_shape_length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_float32(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_float32(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_float32(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else {
				connx_exception("Kernel shape not supported: %u\n", kernel_shape_length);
				return false;
			}
			break;
		case connx_DataType_FLOAT64:
			if(kernel_shape_length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_float64(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_float64(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else if(kernel_shape_length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_float64(Y_base, X_base, X_lengths, kernel_shape, pads, strides, dilations);
					X_base += unit;
				}
			} else {
				connx_exception("Kernel shape not supported: %u\n", kernel_shape_length);
				return false;
			}
			break;
		default:
			abort();
	}

	return true;
}

bool connx_opset_MaxPool_init() {
	connx_Operator_add("MaxPool", 1, 1, 7, MaxPool_resolve, MaxPool_exec,
		connx_DataType_TENSOR_INTEGER32_FLOAT,
		connx_DataType_TENSOR_INTEGER32_FLOAT,
		"auto_pad", connx_DataType_STRING, "NOTSET",
		"ceil_mode", connx_DataType_INT64, 0,
		"dilations", connx_DataType_INT64_ARRAY, 0, NULL, 
		"kernel_shape", connx_DataType_INT64_ARRAY, 0, NULL, 
		"pads", connx_DataType_INT64_ARRAY, 0, NULL, 
		"storage_order", connx_DataType_INT64, 0,
		"strides", connx_DataType_INT64_ARRAY, 0, NULL);

	return true;
}
