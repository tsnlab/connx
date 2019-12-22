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

bool connx_opset_Add_init() {
	connx_Operator_add("Add", 1, 2, 0, Add_resolve, Add_exec,
		connx_DataType_TENSOR_INTEGER32_FLOAT,
		connx_DataType_TENSOR_INTEGER32_FLOAT,
		connx_DataType_TENSOR_INTEGER32_FLOAT);

	return true;
}