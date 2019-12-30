#include <stdlib.h>
#include "opset.h"
#include <connx/connx.h>

#define MINMAX(min, max, a, b)	\
	if((a) > (b)) {		\
		(max) = (a); 	\
		(min) = (b);	\
	} else {			\
		(min) = (b);	\
		(max) = (a);	\
	}

static bool Add_resolve(uintptr_t* stack) {
	connx_Tensor* C = (void*)stack[1];
	connx_Tensor* A = (void*)stack[2];
	connx_Tensor* B = (void*)stack[3];

	int32_t A_idx = A->dimension - C->dimension;
	int32_t B_idx = B->dimension - C->dimension;
	for(uint32_t i = 0; i < C->dimension; i++, A_idx++, B_idx++) {
		uint32_t A_length = A_idx >= 0 ? A->lengths[A_idx] : 0;
		uint32_t B_length = B_idx >= 0 ? B->lengths[B_idx] : 0;
		uint32_t min;
		uint32_t length;
		MINMAX(min, length, A_length, B_length);
		if(min != length) {
			if(min != 0 && min != 1) {
				connx_exception("Broadcasting only supports on dimension 0 or 1 but %u\n", min);
				return false;
			}
		}

		if(C->lengths[i] != length) {
			connx_exception("C's lengths is incorrect: dimension: %u, length: %u != %u", i, C->lengths[i], length);
			return false;
		}
	}

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

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_UINT64:
			{
				uint64_t* a = (uint64_t*)A;
				uint64_t* b = (uint64_t*)B;
				uint64_t* c = (uint64_t*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_INT32:
			{
				int32_t* a = (int32_t*)A;
				int32_t* b = (int32_t*)B;
				int32_t* c = (int32_t*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_INT64:
			{
				int64_t* a = (int64_t*)A;
				int64_t* b = (int64_t*)B;
				int64_t* c = (int64_t*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_FLOAT16:
			{
				float* a = (float*)A;
				float* b = (float*)B;
				float* c = (float*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* a = (float*)A;
				float* b = (float*)B;
				float* c = (float*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* a = (double*)A;
				double* b = (double*)B;
				double* c = (double*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
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
		uint32_t total = 1;
		for(uint32_t i = 0; i < A_dimension; i++) {
			if(A_lengths[i] != B_lengths[i]) {
				goto broadcast;
			}

			total *= A_lengths[i];
		}

		return Add_normal(type, total, C, A, B);
	}

broadcast:

	if(A_dimension == 1 && B_dimension == 1) {
		return Add_leaf(type, C_lengths[0], C, A_lengths[0], A, B_lengths[0], B);
	}

	INIT(A, type)
	INIT(B, type)
	INIT(C, type)

	if(A_dimension == B_dimension) {
		FOR(C, A, B) {
			bool result = Add_broadcast(type, 
					C_dimension - 1, C_lengths + 1, BASE(C, C_idx), 
					A_dimension - 1, A_lengths + 1, BASE(A, A_idx), 
					B_dimension - 1, B_lengths + 1, BASE(B, B_idx));

			if(!result)
				return false;
		}
	} else if(A_dimension > B_dimension) {
		for(uint32_t i = 0; i < C_length; i++) {
			bool result = Add_broadcast(type, 
					C_dimension - 1, C_lengths + 1, BASE(C, i), 
					A_dimension - 1, A_lengths + 1, BASE(A, i), 
					B_dimension, B_lengths, B);

			if(!result)
				return false;
		}
	} else {
		for(uint32_t i = 0; i < C_length; i++) {
			bool result = Add_broadcast(type, 
					C_dimension - 1, C_lengths + 1, BASE(C, i), 
					A_dimension, A_lengths, A, 
					B_dimension - 1, B_lengths + 1, BASE(B, i));

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
