#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <connx/connx.h>

static bool Add_resolve(uintptr_t* stack) {
	connx_Tensor* C = (void*)stack[1];
	connx_Tensor* A = (void*)stack[2];
	connx_Tensor* B = (void*)stack[3];

	if(A->elemType != B->elemType) {
		char buf1[32];
		connx_DataType_toString(A->elemType, 32, buf1);
		char buf2[32];
		connx_DataType_toString(B->elemType, 32, buf2);
		connx_exception("A and B's type is differ: %s vs %s", buf1, buf2);

		return false;
	}

	if(A->elemType != C->elemType) {
		char buf1[32];
		connx_DataType_toString(A->elemType, 32, buf1);
		char buf2[32];
		connx_DataType_toString(C->elemType, 32, buf2);
		connx_exception("A and C's type is differ: %s vs %s", buf1, buf2);

		return false;
	}

	return true;
}

static bool Add_normal(connx_DataType type, uint32_t total, void* C, void* A, void* B) {
	switch(type) {
		case connx_DataType_UINT32:
			{
				uint32_t* a = (uint32_t*)A;
				uint32_t* b = (uint32_t*)B;
				uint32_t* c = (uint32_t*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_DataType_UINT64:
			{
				uint64_t* a = (uint64_t*)A;
				uint64_t* b = (uint64_t*)B;
				uint64_t* c = (uint64_t*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_DataType_INT32:
			{
				int32_t* a = (int32_t*)A;
				int32_t* b = (int32_t*)B;
				int32_t* c = (int32_t*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_DataType_INT64:
			{
				int64_t* a = (int64_t*)A;
				int64_t* b = (int64_t*)B;
				int64_t* c = (int64_t*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_DataType_FLOAT16:
			{
				float* a = (float*)A;
				float* b = (float*)B;
				float* c = (float*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* a = (float*)A;
				float* b = (float*)B;
				float* c = (float*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* a = (double*)A;
				double* b = (double*)B;
				double* c = (double*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		default:
			abort();
	}

	return true;
}

static bool Add_leaf(connx_DataType type, uint32_t C_length, void* C, uint32_t A_length, void* A, uint32_t B_length, void* B) {
	switch(type) {
		case connx_DataType_UINT32:
			{
				uint32_t* a = (uint32_t*)A;
				uint32_t* b = (uint32_t*)B;
				uint32_t* c = (uint32_t*)C;

				for(uint32_t i = 0, A_idx = 0, B_idx = 0; i < C_length; i++, (A_idx = (A_idx + 1) % A_length), (B_idx = (B_idx + 1) % B_length)) {
					c[i] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_UINT64:
			{
				uint64_t* a = (uint64_t*)A;
				uint64_t* b = (uint64_t*)B;
				uint64_t* c = (uint64_t*)C;

				for(uint32_t i = 0, A_idx = 0, B_idx = 0; i < C_length; i++, (A_idx = (A_idx + 1) % A_length), (B_idx = (B_idx + 1) % B_length)) {
					c[i] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_INT32:
			{
				int32_t* a = (int32_t*)A;
				int32_t* b = (int32_t*)B;
				int32_t* c = (int32_t*)C;

				for(uint32_t i = 0, A_idx = 0, B_idx = 0; i < C_length; i++, (A_idx = (A_idx + 1) % A_length), (B_idx = (B_idx + 1) % B_length)) {
					c[i] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_INT64:
			{
				int64_t* a = (int64_t*)A;
				int64_t* b = (int64_t*)B;
				int64_t* c = (int64_t*)C;

				for(uint32_t i = 0, A_idx = 0, B_idx = 0; i < C_length; i++, (A_idx = (A_idx + 1) % A_length), (B_idx = (B_idx + 1) % B_length)) {
					c[i] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_FLOAT16:
			{
				float* a = (float*)A;
				float* b = (float*)B;
				float* c = (float*)C;

				for(uint32_t i = 0, A_idx = 0, B_idx = 0; i < C_length; i++, (A_idx = (A_idx + 1) % A_length), (B_idx = (B_idx + 1) % B_length)) {
					c[i] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* a = (float*)A;
				float* b = (float*)B;
				float* c = (float*)C;

				for(uint32_t i = 0, A_idx = 0, B_idx = 0; i < C_length; i++, (A_idx = (A_idx + 1) % A_length), (B_idx = (B_idx + 1) % B_length)) {
					c[i] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* a = (double*)A;
				double* b = (double*)B;
				double* c = (double*)C;

				for(uint32_t i = 0, A_idx = 0, B_idx = 0; i < C_length; i++, (A_idx = (A_idx + 1) % A_length), (B_idx = (B_idx + 1) % B_length)) {
					c[i] = a[A_idx] + b[B_idx];
				}
			}
			break;
		default:
			abort();
	}

	return true;
}

static bool Add_broadcast(connx_DataType type, uint32_t C_dimension, uint32_t* C_lengths, void* C, uint32_t A_dimension, uint32_t* A_lengths, void* A, uint32_t B_dimension, uint32_t* B_lengths, void* B) {
	if(A_dimension == B_dimension) {
		uint32_t total = 0;
		for(uint32_t i = 0; i < A_dimension; i++) {
			if(A_lengths[i] != B_lengths[i]) {
				goto broadcast;
			}

			total += A_lengths[i];
		}

		return Add_normal(type, total, C, A, B);
	}

broadcast:

	if(A_dimension == 1 && B_dimension == 1) {
		return Add_leaf(type, C_lengths[0], C, A_lengths[0], A, B_lengths[0], B);
	}

	uint32_t dataSize = connx_DataType_size(type);

	uint32_t A_unit = 1;
	for(uint32_t i = 1; i < A_dimension; i++) {
		A_unit *= A_lengths[i];
	}
	A_unit *= dataSize;

	uint32_t B_unit = 1;
	for(uint32_t i = 1; i < B_dimension; i++) {
		B_unit *= B_lengths[i];
	}
	B_unit *= dataSize;

	uint32_t C_unit = 1;
	for(uint32_t i = 1; i < C_dimension; i++) {
		C_unit *= C_lengths[i];
	}
	C_unit *= dataSize;


	if(A_dimension == B_dimension) {
		for(uint32_t i = 0, A_idx = 0, B_idx = 0; i < C_lengths[0]; i++, A_idx = (A_idx + 1) % A_lengths[0], B_idx = (B_idx + 1) % B_lengths[0]) {
			bool result = Add_broadcast(type, C_dimension - 1, C_lengths + 1, C + i * C_unit, A_dimension - 1, A_lengths + 1, A + A_idx * A_unit, B_dimension - 1, B_lengths + 1, B + B_idx * B_unit);
			if(!result)
				return false;
		}
	} else if(A_dimension > B_dimension) {
		for(uint32_t i = 0; i < C_lengths[0]; i++) {
			bool result = Add_broadcast(type, C_dimension - 1, C_lengths + 1, C + i * C_unit, A_dimension - 1, A_lengths + 1, A + i * A_unit, B_dimension, B_lengths, B);
			if(!result)
				return false;
		}
	} else {
		for(uint32_t i = 0; i < C_lengths[0]; i++) {
			bool result = Add_broadcast(type, C_dimension - 1, C_lengths + 1, C + i * C_unit, A_dimension, A_lengths, A, B_dimension - 1, B_lengths + 1, B + i * B_unit);
			if(!result)
				return false;
		}
	}

	return true;
}

static bool Add_exec(uintptr_t* stack) {
	connx_Tensor* C = (void*)stack[1];
	connx_Tensor* A = (void*)stack[2];
	connx_Tensor* B = (void*)stack[3];

	if(connx_Tensor_isShapeEquals(A, B)) {
		return Add_normal(C->elemType, connx_Tensor_total(C), C->base, A->base, B->base);
	} else {
		return Add_broadcast(C->elemType, C->dimension, C->lengths, C->base, A->dimension, A->lengths, A->base, B->dimension, B->lengths, B->base);
	}
}

static bool Relu_resolve(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];

	if(!connx_Tensor_isShapeEquals(X, Y)) {
		char buf1[32];
		char buf2[32];
		connx_Tensor_toShapeString(X, 32, buf1);
		connx_Tensor_toShapeString(Y, 32, buf2);
		connx_exception("X and Y's shape is different: %s vs %s", buf1, buf2);
		return false;
	}

	return true;
}

static bool Relu_exec(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];

	uint32_t total = connx_Tensor_total(X);

	switch(X->elemType) {
		case connx_DataType_FLOAT16:
			{
				float* x = (float*)X->base;
				float* y = (float*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = x[i] > 0 ? x[i] : 0;
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* x = (float*)X->base;
				float* y = (float*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = x[i] > 0 ? x[i] : 0;
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* x = (double*)X->base;
				double* y = (double*)Y->base;

				for(uint32_t i = 0; i < total; i++) {
					y[i] = x[i] > 0 ? x[i] : 0;
				}
			}
			break;
		default:
			abort();
	}

	return true;
}

static bool Reshape_resolve(uintptr_t* stack) {
	return true;
}

static bool Reshape_exec(uintptr_t* stack) {
	connx_Tensor* reshaped = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];
	connx_Tensor* shape = (void*)stack[3];

	uint32_t dimension = 1;
	for(uint32_t i = 0; i < shape->dimension; i++) {
		dimension *= shape->lengths[i];
	}

	uint32_t len = 1;
	int32_t guess = -1;
	uint32_t lengths[dimension];
	int64_t* s = (int64_t*)shape->base;
	for(uint32_t i = 0; i < dimension; i++) {
		if(s[i] == 0) {
			if(i < data->dimension) {
				lengths[i] = data->lengths[i];
			} else {
				connx_exception("zero index cannot be set out of bounds: %ld", i);
				return false;
			}
			len *= lengths[i];
		} else if(s[i] < 0) {
			if(guess == -1) {
				guess = i;
			} else {
				connx_exception("-1 index cannot be repeated more than once: %ld", i);
				return false;
			}
		} else {
			lengths[i] = s[i];
			len *= lengths[i];
		}
	}

	uint32_t total = connx_Tensor_total(data);

	if(guess >= 0) {
		lengths[guess] = total / len;
		len = total;
	}

	if(total != len) {
		connx_exception("shape is not maching: data(%u) vs shape(%u)", total, len);
		return false;
	}

	memcpy(reshaped->base, data->base, connx_DataType_size(data->elemType) * len);

	return true;
}

static bool MatMul_resolve(uintptr_t* stack) {
	connx_Tensor* C = (void*)stack[1];
	connx_Tensor* A = (void*)stack[2];
	connx_Tensor* B = (void*)stack[3];

	if(A->elemType != B->elemType) {
		char buf1[32];
		connx_DataType_toString(A->elemType, 32, buf1);
		char buf2[32];
		connx_DataType_toString(B->elemType, 32, buf2);
		connx_exception("Tensor A and B's type are different: %s vs %s", buf1, buf2);
		return false;
	}

	if(A->elemType != C->elemType) {
		char buf1[32];
		connx_DataType_toString(A->elemType, 32, buf1);
		char buf2[32];
		connx_DataType_toString(C->elemType, 32, buf2);
		connx_exception("Tensor A and C's type are different: %s vs %s", buf1, buf2);
		return false;
	}

	if(A->dimension < 2) {
		connx_exception("A is not a matrix: dimension: %u", A->dimension);
		return false;
	}

	if(A->dimension != B->dimension) {
		connx_exception("A and B's dimension is not equal A's dimension: %u, B's dimension: %u", A->dimension, B->dimension);
		return false;
	}

	if(A->dimension != C->dimension) {
		connx_exception("A and C's dimension is not equal A's dimension: %u, C's dimension: %u", A->dimension, C->dimension);
		return false;
	}

	for(uint32_t i = 0; i < A->dimension - 2; i++) {
		if(A->lengths[i] != B->lengths[i]) {
			connx_exception("Tensor A's %uth length: %u is different to B's %uth length: %u", i, A->lengths[i], i, B->lengths[i]);
			return false;
		}
		if(A->lengths[i] != C->lengths[i]) {
			connx_exception("Tensor A's %uth length: %u is different to C's %uth length: %u", i, A->lengths[i], i, C->lengths[i]);
			return false;
		}
	}

	if(A->lengths[A->dimension - 1] != B->lengths[B->dimension - 2]) {
		char buf1[32];
		connx_Tensor_toShapeString(A, 32, buf1);
		char buf2[32];
		connx_Tensor_toShapeString(B, 32, buf2);
		connx_exception("A's shape and B's shape not matched: %s vs %s", buf1, buf2);
		return false;
	}

	if(A->lengths[A->dimension - 2] != C->lengths[C->dimension - 2]) {
		char buf1[32];
		connx_Tensor_toShapeString(A, 32, buf1);
		char buf2[32];
		connx_Tensor_toShapeString(C, 32, buf2);
		connx_exception("A's shape and C's shape not matched: %s vs %s", buf1, buf2);
		return false;
	}

	if(B->lengths[B->dimension - 1] != C->lengths[C->dimension - 1]) {
		char buf1[32];
		connx_Tensor_toShapeString(B, 32, buf1);
		char buf2[32];
		connx_Tensor_toShapeString(C, 32, buf2);
		connx_exception("B's shape and C's shape not matched: %s vs %s", buf1, buf2);
		return false;
	}

	return true;
}

static bool MatMul_exec(uintptr_t* stack) {
	connx_Tensor* C = (void*)stack[1];
	connx_Tensor* A = (void*)stack[2];
	connx_Tensor* B = (void*)stack[3];

	uint32_t count = 1;	// matrix count
	for(uint32_t i = 0; i < A->dimension - 2; i++) {
		count *= A->lengths[i];
	}

	uint32_t rows = A->lengths[A->dimension - 2];	// matrix row count
	uint32_t cols = B->lengths[B->dimension - 1];	// matrix col count
	uint32_t len = A->lengths[A->dimension - 1];	// A's row or B's col

	switch(A->elemType) {
		case connx_DataType_UINT32:
			{
				uint32_t* a = (uint32_t*)A->base;
				uint32_t* b = (uint32_t*)B->base;
				uint32_t* c = (uint32_t*)C->base;
				uint32_t tmp;

				for(uint32_t i = 0; i < count; i++) {
					for(uint32_t row = 0; row < rows; row++) {
						for(uint32_t col = 0; col < cols; col++) {
							tmp = 0;
							for(uint32_t j = 0; j < len; j++) {
								tmp += a[row * len + j] * b[j * cols + col];
							}
							*c++ = tmp;
						}
					}

					a += len * rows;
					b += len * cols;
				}
			}
			break;
		case connx_DataType_UINT64:
			{
				uint64_t* a = (uint64_t*)A->base;
				uint64_t* b = (uint64_t*)B->base;
				uint64_t* c = (uint64_t*)C->base;
				uint64_t tmp;

				for(uint32_t i = 0; i < count; i++) {
					for(uint32_t row = 0; row < rows; row++) {
						for(uint32_t col = 0; col < cols; col++) {
							tmp = 0;
							for(uint32_t j = 0; j < len; j++) {
								tmp += a[row * len + j] * b[j * cols + col];
							}
							*c++ = tmp;
						}
					}

					a += len * rows;
					b += len * cols;
				}
			}
			break;
		case connx_DataType_INT32:
			{
				int32_t* a = (int32_t*)A->base;
				int32_t* b = (int32_t*)B->base;
				int32_t* c = (int32_t*)C->base;
				int32_t tmp;

				for(uint32_t i = 0; i < count; i++) {
					for(uint32_t row = 0; row < rows; row++) {
						for(uint32_t col = 0; col < cols; col++) {
							tmp = 0;
							for(uint32_t j = 0; j < len; j++) {
								tmp += a[row * len + j] * b[j * cols + col];
							}
							*c++ = tmp;
						}
					}

					a += len * rows;
					b += len * cols;
				}
			}
			break;
		case connx_DataType_INT64:
			{
				int64_t* a = (int64_t*)A->base;
				int64_t* b = (int64_t*)B->base;
				int64_t* c = (int64_t*)C->base;
				int64_t tmp;

				for(uint32_t i = 0; i < count; i++) {
					for(uint32_t row = 0; row < rows; row++) {
						for(uint32_t col = 0; col < cols; col++) {
							tmp = 0;
							for(uint32_t j = 0; j < len; j++) {
								tmp += a[row * len + j] * b[j * cols + col];
							}
							*c++ = tmp;
						}
					}

					a += len * rows;
					b += len * cols;
				}
			}
			break;
		case connx_DataType_FLOAT16:
			{
				float* a = (float*)A->base;
				float* b = (float*)B->base;
				float* c = (float*)C->base;
				float tmp;

				for(uint32_t i = 0; i < count; i++) {
					for(uint32_t row = 0; row < rows; row++) {
						for(uint32_t col = 0; col < cols; col++) {
							tmp = 0;
							for(uint32_t j = 0; j < len; j++) {
								tmp += a[row * len + j] * b[j * cols + col];
							}
							*c++ = tmp;
						}
					}

					a += len * rows;
					b += len * cols;
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* a = (float*)A->base;
				float* b = (float*)B->base;
				float* c = (float*)C->base;
				float tmp;

				for(uint32_t i = 0; i < count; i++) {
					for(uint32_t row = 0; row < rows; row++) {
						for(uint32_t col = 0; col < cols; col++) {
							tmp = 0;
							for(uint32_t j = 0; j < len; j++) {
								tmp += a[row * len + j] * b[j * cols + col];
							}
							*c++ = tmp;
						}
					}

					a += len * rows;
					b += len * cols;
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* a = (double*)A->base;
				double* b = (double*)B->base;
				double* c = (double*)C->base;
				double tmp;

				for(uint32_t i = 0; i < count; i++) {
					for(uint32_t row = 0; row < rows; row++) {
						for(uint32_t col = 0; col < cols; col++) {
							tmp = 0;
							for(uint32_t j = 0; j < len; j++) {
								tmp += a[row * len + j] * b[j * cols + col];
							}
							*c++ = tmp;
						}
					}

					a += len * rows;
					b += len * cols;
				}
			}
			break;
		default:
			abort();
	}

	return true;
}

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

static bool Conv_resolve(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	connx_Tensor* W = (void*)stack[3];
	void* attr_auto_pad = (void*)stack[4];
	void* attr_dilations = (void*)stack[5];
	void* attr_group = (void*)stack[6];
	void* attr_kernel_shape = (void*)stack[7];
	void* attr_pads = (void*)stack[8];
	void* attr_strides = (void*)stack[9];

	char* auto_pad = attr_auto_pad;
	uint32_t dilations_length = connx_Attribute_length(attr_dilations);
	int64_t* group = attr_group;
	int64_t* kernel_shape = connx_Attribute_base(attr_kernel_shape);
	uint32_t kernel_shape_length = connx_Attribute_length(attr_kernel_shape);
	int64_t* pads = connx_Attribute_base(attr_pads);
	uint32_t pads_length = connx_Attribute_length(attr_pads);
	uint32_t strides_length = connx_Attribute_length(attr_strides);

	if(auto_pad[0] == 'S') {	// SAME_UPPER, SAME_LOWER
		int64_t array[kernel_shape_length * 2];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			if(kernel_shape[i] > 2) {
				array[i] = array[i + kernel_shape_length] = (kernel_shape[i] - 1) / 2;
			} else {
				array[i] = array[i + kernel_shape_length] = X->lengths[i];;
			}
		}

		connx_Attribute_delete(attr_pads);
		stack[8] = connx_Attribute_create_ints(kernel_shape_length * 2, array);
		attr_pads = (void*)stack[8];
		pads = connx_Attribute_base(attr_pads);
		pads_length = connx_Attribute_length(attr_pads);
	}

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

	if(kernel_shape_length * 2 != pads_length) {
		connx_exception("pads shape is not maching: kernel_shape's dimension: %u but pads dimension: %u", kernel_shape_length, pads_length);
		return false;
	}

	if(X->dimension != kernel_shape_length + 2) {
		connx_exception("X's dimension: %u is and kernel_shape's dimension: %u is not matching", X->dimension, kernel_shape_length);
		return false;
	}

	if(W->dimension != kernel_shape_length + 2) {
		connx_exception("W's dimension: %u is and kernel_shape's dimension: %u is not matching", W->dimension, kernel_shape_length);
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

	if(Y->lengths[0] != X->lengths[0]) {
		connx_exception("Y's batch size is not matching: Y: %u but X: %u", Y->lengths[0], X->lengths[0]);
		return false;
	}

	if(Y->lengths[1] != W->lengths[0] * *group) {
		connx_exception("Y's feature size is not matching: Y: %u but W: %u", Y->lengths[1], W->lengths[1]);
		return false;
	}

	for(uint32_t i = 0; i < kernel_shape_length; i++) {
		if(Y->lengths[i + 2] != X->lengths[i + 2] - (kernel_shape[i] - 1) + (pads[i] + pads[i + kernel_shape_length])) {
			connx_exception("Y's %uth shape is not matching: Y: %u, expected: %u", i + 2,
					Y->lengths[i + 2],
					X->lengths[i + 2] - (kernel_shape[i] - 1) + (pads[i] + pads[i + kernel_shape_length]));
			return false;
		}
	}

	return true;
}

static void _conv2d_float(__attribute__((unused)) uint32_t* Y_lengths, float* Y, uint32_t* X_lengths, float* X, uint32_t* W_lengths, float* W, int64_t* kernels, int64_t* pads, int64_t* strides) {
	for(int64_t y = -pads[0]; y <= (int64_t)X_lengths[0] + pads[0 + 2] - kernels[0]; y += strides[0]) {
		for(int64_t x = -pads[1]; x <= (int64_t)X_lengths[1] + pads[1 + 2] - kernels[1]; x += strides[1]) {
			float tmp = 0;

			for(uint32_t ky = 0; ky < kernels[0]; ky++) {
				for(uint32_t kx = 0; kx < kernels[1]; kx++) {
					int64_t y2 = y + ky;
					int64_t x2 = x + kx;

					if(y2 >= 0 && x2 >= 0 && y2 < X_lengths[0] && x2 < X_lengths[1]) {
						tmp += X[y2 * X_lengths[0] + x2] * W[ky * W_lengths[0] + kx];
					}
				}
			}

			*Y++ += tmp;
		}
	}
}

static bool Conv_exec(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	connx_Tensor* W = (void*)stack[3];
	__attribute__((unused)) char* attr_auto_pad = (void*)stack[4];
	void* attr_dilations = (void*)stack[5];
	void* attr_group = (void*)stack[6];
	void* attr_kernel_shape = (void*)stack[7];
	void* attr_pads = (void*)stack[8];
	void* attr_strides = (void*)stack[9];

	__attribute__((unused)) uint32_t dilations_length = connx_Attribute_length(attr_dilations);
	int64_t* group = attr_group;
	int64_t* kernel_shape = connx_Attribute_base(attr_kernel_shape);
	uint32_t kernel_shape_length = connx_Attribute_length(attr_kernel_shape);
	int64_t* pads = connx_Attribute_base(attr_pads);
	__attribute__((unused)) uint32_t pads_length = connx_Attribute_length(attr_pads);
	int64_t* strides = connx_Attribute_base(attr_strides);
	__attribute__((unused)) uint32_t strides_length = connx_Attribute_length(attr_strides);

	// make output tensor
	uint32_t x_unit = 1;
	uint32_t y_unit = 1;
	uint32_t w_unit = 1;
	for(uint32_t i = 0; i < kernel_shape_length; i++) {
		y_unit *= Y->lengths[Y->dimension - 2 + i];
		x_unit *= X->lengths[X->dimension - 2 + i];
		w_unit *= W->lengths[W->dimension - 2 + i];
	}

	switch(X->elemType) {
		case connx_DataType_FLOAT16:
		case connx_DataType_FLOAT32:
			{
				float* y_array = (float*)Y->base;

				for(uint32_t batch = 0; batch < X->lengths[0]; batch++) {
					for(uint32_t feature = 0; feature < W->lengths[0]; feature++) {
						for(uint32_t channel = 0; channel < W->lengths[1] * *group; channel++) {
							float* x_array = (float*)X->base + batch * X->lengths[1] * x_unit + channel * x_unit;
							float* w_array = (float*)W->base + feature * W->lengths[1] * w_unit + channel / *group * w_unit;
							_conv2d_float(Y->lengths + Y->dimension - 2, y_array, 
									X->lengths + X->dimension - 2, x_array, 
									W->lengths + W->dimension - 2, w_array, 
									kernel_shape, pads, strides);

						}

						y_array += y_unit;
					}
				}
			}
			break;
		case connx_DataType_FLOAT64:
			break;
		default:
			;
	}

	return true;
}

#define OPERATOR_COUNT	6

uint32_t connx_operator_count;
connx_Operator connx_operators[OPERATOR_COUNT];

bool connx_init() {
	connx_Operator_add("Add", 1, 2, 0, Add_resolve, Add_exec,
		connx_DataType_TENSOR_INTEGER32_FLOAT,
		connx_DataType_TENSOR_INTEGER32_FLOAT,
		connx_DataType_TENSOR_INTEGER32_FLOAT);

	connx_Operator_add("Relu", 1, 1, 0, Relu_resolve, Relu_exec,
		connx_DataType_TENSOR_FLOAT,
		connx_DataType_TENSOR_FLOAT);

	connx_Operator_add("Reshape", 1, 2, 0, Reshape_resolve, Reshape_exec,
		connx_DataType_TENSOR_NUMBER,
		connx_DataType_TENSOR_NUMBER,
		connx_DataType_TENSOR_INT64);

	connx_Operator_add("MatMul", 1, 2, 0, MatMul_resolve, MatMul_exec,
		connx_DataType_TENSOR_INTEGER32_FLOAT,
		connx_DataType_TENSOR_INTEGER32_FLOAT,
		connx_DataType_TENSOR_INTEGER32_FLOAT);

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

	connx_Operator_add("Conv", 1, 2, 6, Conv_resolve, Conv_exec,
		connx_DataType_TENSOR_FLOAT,
		connx_DataType_TENSOR_FLOAT,
		connx_DataType_TENSOR_FLOAT,
		"auto_pad", connx_DataType_STRING, "NOTSET",
		"dilations", connx_DataType_INT64_ARRAY, 0, NULL, 
		"group", connx_DataType_INT64, 1,
		"kernel_shape", connx_DataType_INT64_ARRAY, 0, NULL, 
		"pads", connx_DataType_INT64_ARRAY, 0, NULL, 
		"strides", connx_DataType_INT64_ARRAY, 0, NULL);

	return true;
}
