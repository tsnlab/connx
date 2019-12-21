#include <stdlib.h>
#include <connx/connx.h>

static bool MaxPool_resolve(uintptr_t* stack) {
	__attribute__((unused)) connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	__attribute__((unused)) char* attr_auto_pad = (void*)stack[3];
	__attribute__((unused)) void* attr_ceil_mode = (void*)stack[4];
	void* attr_dilations = (void*)stack[5];
	void* attr_kernel_shape = (void*)stack[6];
	void* attr_pads = (void*)stack[7];
	__attribute__((unused)) void* attr_storage_order = (void*)stack[8];
	void* attr_strides = (void*)stack[9];

	uint32_t kernel_shape_length = connx_Attribute_length(attr_kernel_shape);
	uint32_t dilations_length = connx_Attribute_length(attr_dilations);
	uint32_t strides_length = connx_Attribute_length(attr_strides);
	uint32_t pads_length = connx_Attribute_length(attr_pads);

	if(dilations_length == 0) {
		int64_t array[kernel_shape_length];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			array[i] = 0;
		}

		connx_Attribute_delete(attr_dilations);
		stack[5] = connx_Attribute_create_ints(kernel_shape_length, array);
		attr_dilations = (void*)stack[5];
		dilations_length = connx_Attribute_length(attr_dilations);
	}

	if(pads_length == 0) {
		int64_t array[kernel_shape_length * 2];
		for(uint32_t i = 0; i < kernel_shape_length * 2; i++) {
			array[i] = 0;
		}

		connx_Attribute_delete(attr_pads);
		stack[7] = connx_Attribute_create_ints(kernel_shape_length * 2, array);
		attr_pads = (void*)stack[7];
		pads_length = connx_Attribute_length(attr_pads);
	}

	if(X->dimension < kernel_shape_length) {
		connx_exception("X's dimension: %u is smaller than kernel_shape's dimension: %u", X->dimension, kernel_shape_length);
		return false;
	}

	if(kernel_shape_length != 2 && kernel_shape_length != 3) {
		connx_exception("kernel_shape count must be 2 or 3 but %u", kernel_shape_length);
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
	uint32_t total = 1;
	for(uint32_t i = 0, j = 0; i < X->dimension; i++) {
		if(i >= X->dimension - kernel_shape_length) {
			// Ref: output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
			unit *= X_lengths[j];
			j++;
		}

		total *= X->lengths[i];
	}

	switch(X->elemType) {
		case connx_DataType_UINT32:
			{
				uint32_t* x_array = (uint32_t*)X->base;
				uint32_t* x_end = x_array + total;
				uint32_t* y_array = (uint32_t*)Y->base;
				uint32_t tmp;
				uint32_t tmp2;

				if(kernel_shape_length == 2) {
					while(x_array < x_end) {
						for(int64_t y = -pads[0]; y + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; y += strides[0]) {
							for(int64_t x = -pads[1]; x + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; x += strides[1]) {
								// pool
								tmp = 0;
								for(int64_t j = 0; j < kernel_shape[0]; j++) {
									for(int64_t i = 0; i < kernel_shape[1]; i++) {
										int64_t x2 = x + i;
										int64_t y2 = y + j;
										if(x2 >= 0 && x2 < X_lengths[1] && y2 >= 0 && y2 < X_lengths[0]) {
											tmp2 = x_array[y2 * X_lengths[0] + x2];
											if(tmp2 > tmp)
												tmp = tmp2;
										}
									}
								}
								*y_array++ = tmp;
							}
						}

						x_array += unit;
					}
				} else if(kernel_shape_length == 3) {
					while(x_array < x_end) {
						for(int64_t z = -pads[0]; z + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; z += strides[0]) {
							for(int64_t y = -pads[1]; y + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; y += strides[1]) {
								for(int64_t x = -pads[2]; x + kernel_shape[2] <= X_lengths[2] + pads[2 + 2]; x += strides[2]) {
									// pool
									tmp = 0;
									for(int64_t k = 0; k < kernel_shape[0]; k++) {
										for(int64_t j = 0; j < kernel_shape[1]; j++) {
											for(int64_t i = 0; i < kernel_shape[2]; i++) {
												int64_t x2 = x + i;
												int64_t y2 = y + j;
												int64_t z2 = z + k;
												if(x2 >= 0 && x2 < X_lengths[2] && y2 >= 0 && y2 < X_lengths[1] && z2 >= 0 && z2 < X_lengths[0]) {
													tmp2 = x_array[z2 * X_lengths[0] * X_lengths[1] + y2 * X_lengths[1] + x2];
													if(tmp2 > tmp)
														tmp = tmp2;
												}
											}
										}
									}
									*y_array++ = tmp;
								}
							}
						}

						x_array += unit;
					}
				}
			}
			break;
		case connx_DataType_UINT64:
			{
				uint64_t* x_array = (uint64_t*)X->base;
				uint64_t* x_end = x_array + total;
				uint64_t* y_array = (uint64_t*)Y->base;
				uint64_t tmp;
				uint64_t tmp2;

				if(kernel_shape_length == 2) {
					while(x_array < x_end) {
						for(int64_t y = -pads[0]; y + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; y += strides[0]) {
							for(int64_t x = -pads[1]; x + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; x += strides[1]) {
								// pool
								tmp = 0;
								for(int64_t j = 0; j < kernel_shape[0]; j++) {
									for(int64_t i = 0; i < kernel_shape[1]; i++) {
										int64_t x2 = x + i;
										int64_t y2 = y + j;
										if(x2 >= 0 && x2 < X_lengths[1] && y2 >= 0 && y2 < X_lengths[0]) {
											tmp2 = x_array[y2 * X_lengths[0] + x2];
											if(tmp2 > tmp)
												tmp = tmp2;
										}
									}
								}
								*y_array++ = tmp;
							}
						}

						x_array += unit;
					}
				} else if(kernel_shape_length == 3) {
					while(x_array < x_end) {
						for(int64_t z = -pads[0]; z + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; z += strides[0]) {
							for(int64_t y = -pads[1]; y + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; y += strides[1]) {
								for(int64_t x = -pads[2]; x + kernel_shape[2] <= X_lengths[2] + pads[2 + 2]; x += strides[2]) {
									// pool
									tmp = 0;
									for(int64_t k = 0; k < kernel_shape[0]; k++) {
										for(int64_t j = 0; j < kernel_shape[1]; j++) {
											for(int64_t i = 0; i < kernel_shape[2]; i++) {
												int64_t x2 = x + i;
												int64_t y2 = y + j;
												int64_t z2 = z + k;
												if(x2 >= 0 && x2 < X_lengths[2] && y2 >= 0 && y2 < X_lengths[1] && z2 >= 0 && z2 < X_lengths[0]) {
													tmp2 = x_array[z2 * X_lengths[0] * X_lengths[1] + y2 * X_lengths[1] + x2];
													if(tmp2 > tmp)
														tmp = tmp2;
												}
											}
										}
									}
									*y_array++ = tmp;
								}
							}
						}

						x_array += unit;
					}
				}
			}
			break;
		case connx_DataType_INT32:
			{
				int32_t* x_array = (int32_t*)X->base;
				int32_t* x_end = x_array + total;
				int32_t* y_array = (int32_t*)Y->base;
				int32_t tmp;
				int32_t tmp2;

				if(kernel_shape_length == 2) {
					while(x_array < x_end) {
						for(int64_t y = -pads[0]; y + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; y += strides[0]) {
							for(int64_t x = -pads[1]; x + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; x += strides[1]) {
								// pool
								tmp = 0;
								for(int64_t j = 0; j < kernel_shape[0]; j++) {
									for(int64_t i = 0; i < kernel_shape[1]; i++) {
										int64_t x2 = x + i;
										int64_t y2 = y + j;
										if(x2 >= 0 && x2 < X_lengths[1] && y2 >= 0 && y2 < X_lengths[0]) {
											tmp2 = x_array[y2 * X_lengths[0] + x2];
											if(tmp2 > tmp)
												tmp = tmp2;
										}
									}
								}
								*y_array++ = tmp;
							}
						}

						x_array += unit;
					}
				} else if(kernel_shape_length == 3) {
					while(x_array < x_end) {
						for(int64_t z = -pads[0]; z + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; z += strides[0]) {
							for(int64_t y = -pads[1]; y + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; y += strides[1]) {
								for(int64_t x = -pads[2]; x + kernel_shape[2] <= X_lengths[2] + pads[2 + 2]; x += strides[2]) {
									// pool
									tmp = 0;
									for(int64_t k = 0; k < kernel_shape[0]; k++) {
										for(int64_t j = 0; j < kernel_shape[1]; j++) {
											for(int64_t i = 0; i < kernel_shape[2]; i++) {
												int64_t x2 = x + i;
												int64_t y2 = y + j;
												int64_t z2 = z + k;
												if(x2 >= 0 && x2 < X_lengths[2] && y2 >= 0 && y2 < X_lengths[1] && z2 >= 0 && z2 < X_lengths[0]) {
													tmp2 = x_array[z2 * X_lengths[0] * X_lengths[1] + y2 * X_lengths[1] + x2];
													if(tmp2 > tmp)
														tmp = tmp2;
												}
											}
										}
									}
									*y_array++ = tmp;
								}
							}
						}

						x_array += unit;
					}
				}
			}
			break;
		case connx_DataType_INT64:
			{
				int64_t* x_array = (int64_t*)X->base;
				int64_t* x_end = x_array + total;
				int64_t* y_array = (int64_t*)Y->base;
				int64_t tmp;
				int64_t tmp2;

				if(kernel_shape_length == 2) {
					while(x_array < x_end) {
						for(int64_t y = -pads[0]; y + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; y += strides[0]) {
							for(int64_t x = -pads[1]; x + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; x += strides[1]) {
								// pool
								tmp = 0;
								for(int64_t j = 0; j < kernel_shape[0]; j++) {
									for(int64_t i = 0; i < kernel_shape[1]; i++) {
										int64_t x2 = x + i;
										int64_t y2 = y + j;
										if(x2 >= 0 && x2 < X_lengths[1] && y2 >= 0 && y2 < X_lengths[0]) {
											tmp2 = x_array[y2 * X_lengths[0] + x2];
											if(tmp2 > tmp)
												tmp = tmp2;
										}
									}
								}
								*y_array++ = tmp;
							}
						}

						x_array += unit;
					}
				} else if(kernel_shape_length == 3) {
					while(x_array < x_end) {
						for(int64_t z = -pads[0]; z + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; z += strides[0]) {
							for(int64_t y = -pads[1]; y + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; y += strides[1]) {
								for(int64_t x = -pads[2]; x + kernel_shape[2] <= X_lengths[2] + pads[2 + 2]; x += strides[2]) {
									// pool
									tmp = 0;
									for(int64_t k = 0; k < kernel_shape[0]; k++) {
										for(int64_t j = 0; j < kernel_shape[1]; j++) {
											for(int64_t i = 0; i < kernel_shape[2]; i++) {
												int64_t x2 = x + i;
												int64_t y2 = y + j;
												int64_t z2 = z + k;
												if(x2 >= 0 && x2 < X_lengths[2] && y2 >= 0 && y2 < X_lengths[1] && z2 >= 0 && z2 < X_lengths[0]) {
													tmp2 = x_array[z2 * X_lengths[0] * X_lengths[1] + y2 * X_lengths[1] + x2];
													if(tmp2 > tmp)
														tmp = tmp2;
												}
											}
										}
									}
									*y_array++ = tmp;
								}
							}
						}

						x_array += unit;
					}
				}
			}
			break;
		case connx_DataType_FLOAT16:
		case connx_DataType_FLOAT32:
			{
				float* x_array = (float*)X->base;
				float* x_end = x_array + total;
				float* y_array = (float*)Y->base;
				float tmp;
				float tmp2;

				if(kernel_shape_length == 2) {
					while(x_array < x_end) {
						for(int64_t y = -pads[0]; y + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; y += strides[0]) {
							for(int64_t x = -pads[1]; x + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; x += strides[1]) {
								// pool
								tmp = 0;
								for(int64_t j = 0; j < kernel_shape[0]; j++) {
									for(int64_t i = 0; i < kernel_shape[1]; i++) {
										int64_t x2 = x + i;
										int64_t y2 = y + j;
										if(x2 >= 0 && x2 < X_lengths[1] && y2 >= 0 && y2 < X_lengths[0]) {
											tmp2 = x_array[y2 * X_lengths[0] + x2];
											if(tmp2 > tmp)
												tmp = tmp2;
										}
									}
								}
								*y_array++ = tmp;
							}
						}

						x_array += unit;
					}
				} else if(kernel_shape_length == 3) {
					while(x_array < x_end) {
						for(int64_t z = -pads[0]; z + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; z += strides[0]) {
							for(int64_t y = -pads[1]; y + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; y += strides[1]) {
								for(int64_t x = -pads[2]; x + kernel_shape[2] <= X_lengths[2] + pads[2 + 2]; x += strides[2]) {
									// pool
									tmp = 0;
									for(int64_t k = 0; k < kernel_shape[0]; k++) {
										for(int64_t j = 0; j < kernel_shape[1]; j++) {
											for(int64_t i = 0; i < kernel_shape[2]; i++) {
												int64_t x2 = x + i;
												int64_t y2 = y + j;
												int64_t z2 = z + k;
												if(x2 >= 0 && x2 < X_lengths[2] && y2 >= 0 && y2 < X_lengths[1] && z2 >= 0 && z2 < X_lengths[0]) {
													tmp2 = x_array[z2 * X_lengths[0] * X_lengths[1] + y2 * X_lengths[1] + x2];
													if(tmp2 > tmp)
														tmp = tmp2;
												}
											}
										}
									}
									*y_array++ = tmp;
								}
							}
						}

						x_array += unit;
					}
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* x_array = (double*)X->base;
				double* x_end = x_array + total;
				double* y_array = (double*)Y->base;
				double tmp;
				double tmp2;

				if(kernel_shape_length == 2) {
					while(x_array < x_end) {
						for(int64_t y = -pads[0]; y + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; y += strides[0]) {
							for(int64_t x = -pads[1]; x + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; x += strides[1]) {
								// pool
								tmp = 0;
								for(int64_t j = 0; j < kernel_shape[0]; j++) {
									for(int64_t i = 0; i < kernel_shape[1]; i++) {
										int64_t x2 = x + i;
										int64_t y2 = y + j;
										if(x2 >= 0 && x2 < X_lengths[1] && y2 >= 0 && y2 < X_lengths[0]) {
											tmp2 = x_array[y2 * X_lengths[0] + x2];
											if(tmp2 > tmp)
												tmp = tmp2;
										}
									}
								}
								*y_array++ = tmp;
							}
						}

						x_array += unit;
					}
				} else if(kernel_shape_length == 3) {
					while(x_array < x_end) {
						for(int64_t z = -pads[0]; z + kernel_shape[0] <= X_lengths[0] + pads[0 + 2]; z += strides[0]) {
							for(int64_t y = -pads[1]; y + kernel_shape[1] <= X_lengths[1] + pads[1 + 2]; y += strides[1]) {
								for(int64_t x = -pads[2]; x + kernel_shape[2] <= X_lengths[2] + pads[2 + 2]; x += strides[2]) {
									// pool
									tmp = 0;
									for(int64_t k = 0; k < kernel_shape[0]; k++) {
										for(int64_t j = 0; j < kernel_shape[1]; j++) {
											for(int64_t i = 0; i < kernel_shape[2]; i++) {
												int64_t x2 = x + i;
												int64_t y2 = y + j;
												int64_t z2 = z + k;
												if(x2 >= 0 && x2 < X_lengths[2] && y2 >= 0 && y2 < X_lengths[1] && z2 >= 0 && z2 < X_lengths[0]) {
													tmp2 = x_array[z2 * X_lengths[0] * X_lengths[1] + y2 * X_lengths[1] + x2];
													if(tmp2 > tmp)
														tmp = tmp2;
												}
											}
										}
									}
									*y_array++ = tmp;
								}
							}
						}

						x_array += unit;
					}
				}
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
		"strides", connx_DataType_INT64_ARRAY, 0);

	return true;
}
