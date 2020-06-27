#include <inttypes.h>
#include <stdlib.h>
#include <connx/connx.h>

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
		connx_exception("A is not a matrix: dimension: %" PRIu32, A->dimension);
		return false;
	}

	if(A->dimension != B->dimension) {
		connx_exception("A and B's dimension is not equal A's dimension: %" PRIu32 ", B's dimension: %" PRIu32, A->dimension, B->dimension);
		return false;
	}

	if(A->dimension != C->dimension) {
		connx_exception("A and C's dimension is not equal A's dimension: %" PRIu32 ", C's dimension: %" PRIu32, A->dimension, C->dimension);
		return false;
	}

	for(uint32_t i = 0; i < A->dimension - 2; i++) {
		if(A->lengths[i] != B->lengths[i]) {
			connx_exception("Tensor A's %" PRIu32 "th length: %" PRIu32 " is different to B's %" PRIu32 "th length: %" PRIu32, i, A->lengths[i], i, B->lengths[i]);
			return false;
		}
		if(A->lengths[i] != C->lengths[i]) {
			connx_exception("Tensor A's %" PRIu32 "th length: %" PRIu32 " is different to C's %" PRIu32 "th length: %" PRIu32, i, A->lengths[i], i, C->lengths[i]);
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

bool connx_opset_MatMul_init() {
	connx_Operator_add("MatMul", 1, 2, 0, MatMul_resolve, MatMul_exec,
		connx_DataType_TENSOR_INTEGER32_FLOAT,
		connx_DataType_TENSOR_INTEGER32_FLOAT,
		connx_DataType_TENSOR_INTEGER32_FLOAT);

	return true;
}
