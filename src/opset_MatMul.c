#include <stdio.h>
#include <string.h>
#include <connx/operator.h>
#include <connx/backend.h>

bool opset_MatMul(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* C = CONNX_GET_OUTPUT(0);
	connx_Tensor* A = CONNX_GET_INPUT(0);
	connx_Tensor* B = CONNX_GET_INPUT(1);

	uint32_t count = 1;	// matrix count
	for(uint32_t i = 0; i < A->dimension - 2; i++) {
		count *= A->lengths[i];
	}

	uint32_t rows = A->lengths[A->dimension - 2];	// matrix row count
	uint32_t cols = B->lengths[B->dimension - 1];	// matrix col count
	uint32_t len = A->lengths[A->dimension - 1];	// A's row or B's col

	if(C == NULL) {
		uint32_t lengths[A->dimension];
		memcpy(lengths, A->lengths, sizeof(uint32_t) * (A->dimension - 2));
		lengths[A->dimension - 2] = A->lengths[A->dimension - 2];
		lengths[A->dimension - 1] = B->lengths[B->dimension - 1];

		C = connx_Tensor_create(backend->pal, A->type, A->dimension, lengths);
		CONNX_SET_OUTPUT(0, C);
	}

	switch(A->type) {
		case connx_UINT32:
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
		case connx_UINT64:
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
		case connx_INT32:
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
		case connx_INT64:
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
		case connx_FLOAT16:
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
		case connx_FLOAT32:
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
		case connx_FLOAT64:
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
			{
				backend->pal->error(backend->pal, "Not supported element type: %s\n", connx_DataType_name(A->type));
				return false;
			}
	}

	return true;
}
