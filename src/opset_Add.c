#include <stdio.h>
#include <connx/operator.h>
#include <connx/backend.h>

#define MINMAX(min, max, a, b)	\
	if((a) > (b)) {		\
		(max) = (a); 	\
		(min) = (b);	\
	} else {			\
		(min) = (b);	\
		(max) = (a);	\
	}

#define __unit(dimension, lengths) ({			\
	uint32_t unit = 1;							\
	for(uint32_t i = 1; i < dimension; i++) {	\
		unit *= lengths[i];						\
	}											\
												\
	unit;										\
})

#define INIT(array, type)	\
	uint32_t array##_unit = __unit(array##_dimension, array##_lengths) * connx_DataType_size(type);	\
	uint32_t array##_length = array##_lengths[0];

#define BASE(array, idx) (array + array##_unit * idx)

#define FOR(array0, array1, array2)	\
	for(uint32_t array0##_idx = 0, array1##_idx = 0, array2##_idx = 0;	\
			array0##_idx < array0##_length;								\
			array0##_idx++,		\
			array1##_idx = (array1##_idx + 1) % array1##_length,		\
			array2##_idx = (array2##_idx + 1) % array2##_length)

static bool Add_normal(connx_DataType type, uint32_t total, void* C, void* A, void* B) {
	switch(type) {
		case connx_UINT32:
			{
				uint32_t* a = (uint32_t*)A;
				uint32_t* b = (uint32_t*)B;
				uint32_t* c = (uint32_t*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_UINT64:
			{
				uint64_t* a = (uint64_t*)A;
				uint64_t* b = (uint64_t*)B;
				uint64_t* c = (uint64_t*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_INT32:
			{
				int32_t* a = (int32_t*)A;
				int32_t* b = (int32_t*)B;
				int32_t* c = (int32_t*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_INT64:
			{
				int64_t* a = (int64_t*)A;
				int64_t* b = (int64_t*)B;
				int64_t* c = (int64_t*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_FLOAT16:
			{
				float* a = (float*)A;
				float* b = (float*)B;
				float* c = (float*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_FLOAT32:
			{
				float* a = (float*)A;
				float* b = (float*)B;
				float* c = (float*)C;

				for(uint32_t i = 0; i < total; i++) {
					c[i] = a[i] + b[i];
				}
			}
			break;
		case connx_FLOAT64:
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
			return false;
	}

	return true;
}

static bool Add_leaf(connx_DataType type, uint32_t C_length, void* C, uint32_t A_length, void* A, uint32_t B_length, void* B) {
	switch(type) {
		case connx_UINT32:
			{
				uint32_t* a = (uint32_t*)A;
				uint32_t* b = (uint32_t*)B;
				uint32_t* c = (uint32_t*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_UINT64:
			{
				uint64_t* a = (uint64_t*)A;
				uint64_t* b = (uint64_t*)B;
				uint64_t* c = (uint64_t*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_INT32:
			{
				int32_t* a = (int32_t*)A;
				int32_t* b = (int32_t*)B;
				int32_t* c = (int32_t*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_INT64:
			{
				int64_t* a = (int64_t*)A;
				int64_t* b = (int64_t*)B;
				int64_t* c = (int64_t*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_FLOAT16:
			{
				float* a = (float*)A;
				float* b = (float*)B;
				float* c = (float*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_FLOAT32:
			{
				float* a = (float*)A;
				float* b = (float*)B;
				float* c = (float*)C;

				FOR(C, A, B) {
					c[C_idx] = a[A_idx] + b[B_idx];
				}
			}
			break;
		case connx_FLOAT64:
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
			return false;
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

bool opset_Add(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* C = CONNX_GET_OUTPUT(0);
	connx_Tensor* A = CONNX_GET_INPUT(0);
	connx_Tensor* B = CONNX_GET_INPUT(1);

	if(C == NULL) {
		uint32_t __attribute__((unused)) min, max;
		MINMAX(min, max, A->dimension, B->dimension);

		uint32_t C_dimension = max;
		uint32_t lengths[C_dimension];

		int32_t A_idx = A->dimension - C_dimension;
		int32_t B_idx = B->dimension - C_dimension;
		for(uint32_t i = 0; i < C_dimension; i++, A_idx++, B_idx++) {
			uint32_t A_length = A_idx >= 0 ? A->lengths[A_idx] : 0;
			uint32_t B_length = B_idx >= 0 ? B->lengths[B_idx] : 0;
			MINMAX(min, max, A_length, B_length);

			lengths[i] = max;
		}

		C = connx_Tensor_create(backend->hal, A->type, C_dimension, lengths);
		CONNX_SET_OUTPUT(0, C);
	}

	if(connx_Tensor_is_shape_equals(A, B)) {
		return Add_normal(C->type, connx_Tensor_total(C), C->base, A->base, B->base);
	} else {
		return Add_broadcast(C->type, C->dimension, C->lengths, C->base, A->dimension, A->lengths, A->base, B->dimension, B->lengths, B->base);
	}
}
