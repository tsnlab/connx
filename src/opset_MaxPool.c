#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include <float.h>
#include <connx/operator.h>
#include <connx/backend.h>

static void* pool_1d_uint32(uint32_t* Y, uint32_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	uint32_t tmp, tmp2;

	for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = 0;
		for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int32_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < (int32_t)X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_uint32(uint32_t* Y, uint32_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	uint32_t tmp, tmp2;

	for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = 0;
			for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int32_t t1 = d1 + kd1 + dilations[0] - 1;
					int32_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1]) {
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

static void* pool_3d_uint32(uint32_t* Y, uint32_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	uint32_t tmp, tmp2;

	for(int32_t d3 = -pads[2]; d3 <= (int32_t)X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = 0;
				for(int32_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int32_t t1 = d1 + kd1 + dilations[0] - 1;
							int32_t t2 = d2 + kd2 + dilations[1] - 1;
							int32_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1] && t3 >= 0 && t3 < (int32_t)X_lengths[2]) {
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

static void* pool_1d_uint64(uint64_t* Y, uint64_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	uint64_t tmp, tmp2;

	for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = 0;
		for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int32_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < (int32_t)X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_uint64(uint64_t* Y, uint64_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	uint64_t tmp, tmp2;

	for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = 0;
			for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int32_t t1 = d1 + kd1 + dilations[0] - 1;
					int32_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1]) {
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

static void* pool_3d_uint64(uint64_t* Y, uint64_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	uint64_t tmp, tmp2;

	for(int32_t d3 = -pads[2]; d3 <= (int32_t)X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = 0;
				for(int32_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int32_t t1 = d1 + kd1 + dilations[0] - 1;
							int32_t t2 = d2 + kd2 + dilations[1] - 1;
							int32_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1] && t3 >= 0 && t3 < (int32_t)X_lengths[2]) {
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

static void* pool_1d_int32(int32_t* Y, int32_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	int32_t tmp, tmp2;

	for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = INT32_MIN;
		for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int32_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < (int32_t)X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_int32(int32_t* Y, int32_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	int32_t tmp, tmp2;

	for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = INT32_MIN;
			for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int32_t t1 = d1 + kd1 + dilations[0] - 1;
					int32_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1]) {
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

static void* pool_3d_int32(int32_t* Y, int32_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	int32_t tmp, tmp2;

	for(int32_t d3 = -pads[2]; d3 <= (int32_t)X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = INT32_MIN;
				for(int32_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int32_t t1 = d1 + kd1 + dilations[0] - 1;
							int32_t t2 = d2 + kd2 + dilations[1] - 1;
							int32_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1] && t3 >= 0 && t3 < (int32_t)X_lengths[2]) {
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

static void* pool_1d_int64(int64_t* Y, int64_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	int64_t tmp, tmp2;

	for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = INT32_MIN;
		for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int32_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < (int32_t)X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_int64(int64_t* Y, int64_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	int64_t tmp, tmp2;

	for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = INT32_MIN;
			for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int32_t t1 = d1 + kd1 + dilations[0] - 1;
					int32_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1]) {
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

static void* pool_3d_int64(int64_t* Y, int64_t* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	int64_t tmp, tmp2;

	for(int32_t d3 = -pads[2]; d3 <= (int32_t)X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = INT32_MIN;
				for(int32_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int32_t t1 = d1 + kd1 + dilations[0] - 1;
							int32_t t2 = d2 + kd2 + dilations[1] - 1;
							int32_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1] && t3 >= 0 && t3 < (int32_t)X_lengths[2]) {
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

static void* pool_1d_float32(float* Y, float* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	float tmp, tmp2;

	for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = -FLT_MAX;
		for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int32_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < (int32_t)X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_float32(float* Y, float* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	float tmp, tmp2;

	for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = -FLT_MAX;
			for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int32_t t1 = d1 + kd1 + dilations[0] - 1;
					int32_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1]) {
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

static void* pool_3d_float32(float* Y, float* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	float tmp, tmp2;

	for(int32_t d3 = -pads[2]; d3 <= (int32_t)X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = -FLT_MAX;
				for(int32_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int32_t t1 = d1 + kd1 + dilations[0] - 1;
							int32_t t2 = d2 + kd2 + dilations[1] - 1;
							int32_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1] && t3 >= 0 && t3 < (int32_t)X_lengths[2]) {
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

static void* pool_1d_float64(double* Y, double* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	double tmp, tmp2;

	for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 1] - kernel[0]; d1 += strides[0] * dilations[0]) {
		// pool
		tmp = -FLT_MAX;
		for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
			int32_t t1 = d1 + kd1 + dilations[0] - 1;
			if(t1 >= 0 && t1 < (int32_t)X_lengths[0]) {
				tmp2 = X[t1];
				if(tmp2 > tmp)
					tmp = tmp2;
			}
		}
		*Y++ = tmp;
	}

	return Y;
}

static void* pool_2d_float64(double* Y, double* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	float tmp, tmp2;

	for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 2] - kernel[1]; d2 += strides[1] * dilations[1]) {
		for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 2] - kernel[0]; d1 += strides[0] * dilations[0]) {
			// pool
			tmp = -FLT_MAX;
			for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
				for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
					int32_t t1 = d1 + kd1 + dilations[0] - 1;
					int32_t t2 = d2 + kd2 + dilations[1] - 1;
					if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1]) {
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

static void* pool_3d_float64(double* Y, double* X, uint32_t* X_lengths, int32_t* kernel, int32_t* pads, int32_t* strides, int32_t* dilations) {
	double tmp, tmp2;

	for(int32_t d3 = -pads[2]; d3 <= (int32_t)X_lengths[2] + pads[2 + 3] - kernel[2]; d3 += strides[2] * dilations[2]) {
		for(int32_t d2 = -pads[1]; d2 <= (int32_t)X_lengths[1] + pads[1 + 3] - kernel[1]; d2 += strides[1] * dilations[1]) {
			for(int32_t d1 = -pads[0]; d1 <= (int32_t)X_lengths[0] + pads[0 + 3] - kernel[0]; d1 += strides[0] * dilations[0]) {
				// pool
				tmp = -DBL_MAX;
				for(int32_t kd3 = 0; kd3 < kernel[2]; kd3++) {
					for(int32_t kd2 = 0; kd2 < kernel[1]; kd2++) {
						for(int32_t kd1 = 0; kd1 < kernel[0]; kd1++) {
							int32_t t1 = d1 + kd1 + dilations[0] - 1;
							int32_t t2 = d2 + kd2 + dilations[1] - 1;
							int32_t t3 = d3 + kd3 + dilations[2] - 1;

							if(t1 >= 0 && t1 < (int32_t)X_lengths[0] && t2 >= 0 && t2 < (int32_t)X_lengths[1] && t3 >= 0 && t3 < (int32_t)X_lengths[2]) {
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
bool opset_MaxPool(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	// TODO: implement ceil_mode, storage_order
	// TODO: ref: https://datascience.stackexchange.com/questions/28881/what-is-dilated-pooling-and-how-it-works-mathematically
	//
	// outputs
	connx_Tensor* Y = CONNX_GET_OUTPUT(0);

	// inputs
	connx_Tensor* X = CONNX_GET_INPUT(0);

	// attributes
	connx_AttributeString* auto_pad = CONNX_GET_ATTRIBUTE(0);
	__attribute__((unused)) connx_AttributeInt* ceil_mode = CONNX_GET_ATTRIBUTE(1);
	connx_AttributeInts* dilations = CONNX_GET_ATTRIBUTE(2);
	connx_AttributeInts* kernel_shape = CONNX_GET_ATTRIBUTE(3);
	connx_AttributeInts* pads = CONNX_GET_ATTRIBUTE(4);
	__attribute__((unused)) connx_AttributeInt* storage_order = CONNX_GET_ATTRIBUTE(5);
	connx_AttributeInts* strides = CONNX_GET_ATTRIBUTE(6);

	int32_t pad_values[kernel_shape->length * 2];
	bzero(pad_values, sizeof(int32_t) * kernel_shape->length * 2);

	if(auto_pad->value[0] == 'S') {
		for(uint32_t i = 0; i < kernel_shape->length; i++) {
			// Same logic with Conv
			// VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
			// SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
			// output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
			uint32_t input_shape = X->lengths[X->dimension - kernel_shape->length + i];
			uint32_t output_shape = input_shape / strides->values[i] + (input_shape % strides->values[i] > 0 ? 1 : 0);
			uint32_t pad = (output_shape - 1) * strides->values[i] + ((kernel_shape->values[i] - 1) * dilations->values[i] + 1) - input_shape;
			pad_values[i] = pad_values[i + kernel_shape->length] = pad / 2;
			if(pad % 2 == 1) {
				if(auto_pad->value[5] == 'U') {	// SAME_UPPER
					pad_values[i + kernel_shape->length]++;
				} else {						// SAME_LOWER
					pad_values[i]++;
				}
			}
		}
	} else if(auto_pad->value[0] == 'V') {
		bzero(pad_values, sizeof(int32_t) * kernel_shape->length * 2);
	} else {
		memcpy(pad_values, pads->values, sizeof(int32_t) * kernel_shape->length * 2);
	}

	if(Y == NULL) {
		uint32_t lengths[X->dimension];
		memcpy(lengths, X->lengths, sizeof(uint32_t) * (X->dimension - kernel_shape->length));

		if(ceil_mode->value != 0) {
			// output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)

			for(uint32_t i = 0; i < kernel_shape->length; i++) {
				uint32_t i2 = X->dimension - kernel_shape->length + i;
				lengths[i2] = X->lengths[i2] + pad_values[i] - ((kernel_shape->values[i] - 1) * dilations->values[i] + 1);
				lengths[i2] = lengths[i2] / strides->values[i] + 1 + (lengths[i2] % strides->values[i] > 0 ? 1 : 0);
			}
		} else {
			// output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
			//

			for(uint32_t i = 0; i < kernel_shape->length; i++) {
				uint32_t i2 = X->dimension - kernel_shape->length + i;
				lengths[i2] = (X->lengths[i2] + pad_values[i] - ((kernel_shape->values[i] - 1) * dilations->values[i] + 1)) / strides->values[i] + 1;
			}
		}

		Y = connx_Tensor_create(backend->pal, X->type, X->dimension, lengths);
		CONNX_SET_OUTPUT(0, Y);
	}

	uint32_t* X_lengths = X->lengths + X->dimension - kernel_shape->length;
	uint32_t unit = 1;
	uint32_t loop = 1;
	for(uint32_t i = 0; i < X->dimension; i++) {
		if(i < X->dimension - kernel_shape->length) {
			loop *= X->lengths[i];
		} else {
			unit *= X->lengths[i];
		}
	}

	unit *= connx_DataType_size(X->type);

	void* X_base = X->base;
	void* Y_base = Y->base;

	switch(X->type) {
		case connx_UINT32:
			if(kernel_shape->length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_uint32(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_uint32(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_uint32(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else {
				backend->pal->error(backend->pal, "Kernel shape not supported: %" PRIu32 "\n", kernel_shape->length);
				return false;
			}
			break;
		case connx_UINT64:
			if(kernel_shape->length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_uint64(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_uint64(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_uint64(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else {
				backend->pal->error(backend->pal, "Kernel shape not supported: %" PRIu32 "\n", kernel_shape->length);
				return false;
			}
			break;
		case connx_INT32:
			if(kernel_shape->length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_int32(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_int32(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_int32(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else {
				backend->pal->error(backend->pal, "Kernel shape not supported: %" PRIu32 "\n", kernel_shape->length);
				return false;
			}
			break;
		case connx_INT64:
			if(kernel_shape->length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_int64(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_int64(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_int64(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else {
				backend->pal->error(backend->pal, "Kernel shape not supported: %" PRIu32 "\n", kernel_shape->length);
				return false;
			}
			break;
		case connx_FLOAT16:
		case connx_FLOAT32:
			if(kernel_shape->length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_float32(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_float32(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_float32(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else {
				backend->pal->error(backend->pal, "Kernel shape not supported: %" PRIu32 "\n", kernel_shape->length);
				return false;
			}
			break;
		case connx_FLOAT64:
			if(kernel_shape->length == 1) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_1d_float64(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 2) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_2d_float64(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else if(kernel_shape->length == 3) {
				for(uint32_t i = 0; i < loop; i++) {
					Y_base = pool_3d_float64(Y_base, X_base, X_lengths, kernel_shape->values, pad_values, strides->values, dilations->values);
					X_base += unit;
				}
			} else {
				backend->pal->error(backend->pal, "Kernel shape not supported: %" PRIu32 "\n", kernel_shape->length);
				return false;
			}
			break;
		default:
			backend->pal->error(backend->pal, "Unsupported type: %u\n", X->type);
			return false;
	}

	return true;
}
