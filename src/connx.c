#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <setjmp.h>
#include <stdarg.h>
#include <math.h>
#include <pthread.h>
#include <connx/connx.h>

#define EXCEPTION_MESSAGE_SIZE 128

static char _connx_exception_message[EXCEPTION_MESSAGE_SIZE];

// exception
void connx_exception(char* format, ...) {
	va_list list;

	va_start(list, format);
	int len = vsnprintf(_connx_exception_message, EXCEPTION_MESSAGE_SIZE, format, list);
	va_end(list);

	if(len < EXCEPTION_MESSAGE_SIZE)
		fprintf(stderr, "%s\n", _connx_exception_message);
	else
		fprintf(stderr, "%s...\n", _connx_exception_message);
}

const char* connx_exception_message() {
	return _connx_exception_message;
}

// memory management
void* connx_alloc(size_t size) {
	void* ptr = calloc(1, size);
	if(ptr == NULL) {
		fprintf(stderr, "Not enough memory");
		abort();
	}

	return ptr;
}

void* connx_realloc(void* ptr, size_t size) {
	ptr = realloc(ptr, size);
	if(ptr == NULL) {
		fprintf(stderr, "Not enough memory");
		abort();
	}

	return ptr;
}

void connx_free(void* ptr) {
	free(ptr);
}

// data type
int connx_DataType_toString(connx_DataType type, int len, char* buf) {
	int pos = 0;

	if((type & connx_DataType_ARRAY) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "array");
	}

	if((type & connx_DataType_TENSOR) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "tensor");;
	}

	if((type & connx_DataType_SEQUENCE) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "sequence");
	}

	if((type & connx_DataType_MAP) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "map");
	}

	if((type & connx_DataType_GRAPH) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "graph");
	}

	if((type & connx_DataType_UINT8) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "uint8");
	}

	if((type & connx_DataType_UINT16) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "uint16");
	}

	if((type & connx_DataType_UINT32) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "uint32");
	}

	if((type & connx_DataType_UINT64) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "uint64");
	}

	if((type & connx_DataType_INT8) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "int8");
	}

	if((type & connx_DataType_INT16) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "int16");
	}

	if((type & connx_DataType_INT32) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "int32");
	}

	if((type & connx_DataType_INT64) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "int64");
	}

	if((type & connx_DataType_FLOAT16) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "float16");
	}

	if((type & connx_DataType_FLOAT32) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "float32");
	}

	if((type & connx_DataType_FLOAT64) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "float64");
	}

	if((type & connx_DataType_BOOL) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "bool");
	}

	if((type & connx_DataType_STRING) > 0) {
		if(pos > 0)
			pos += snprintf(buf + pos, len - pos, " | ");
		pos += snprintf(buf + pos, len - pos, "string");
	}

	if(pos == 0) {
		pos += snprintf(buf, len - pos, "void");
	}

	return pos;
}

uint32_t connx_DataType_size(connx_DataType type) {
	switch(type) {
		case connx_DataType_VOID:
			return 0;
		case connx_DataType_UINT8:
			return sizeof(uint8_t);
		case connx_DataType_UINT16:
			return sizeof(uint16_t);
		case connx_DataType_UINT32:
			return sizeof(uint32_t);
		case connx_DataType_UINT64:
			return sizeof(uint64_t);
		case connx_DataType_INT8:
			return sizeof(int8_t);
		case connx_DataType_INT16:
			return sizeof(int16_t);
		case connx_DataType_INT32:
			return sizeof(int32_t);
		case connx_DataType_INT64:
			return sizeof(int64_t);
		case connx_DataType_FLOAT16:
			return sizeof(uint16_t);
		case connx_DataType_FLOAT32:
			return sizeof(float);
		case connx_DataType_FLOAT64:
			return sizeof(double);
		case connx_DataType_BOOL:
			return sizeof(bool);
		case connx_DataType_STRING:
			return sizeof(char*);
		case connx_DataType_TENSOR:
			return sizeof(connx_Tensor);
		case connx_DataType_SEQUENCE:
			return sizeof(connx_Sequence);
		case connx_DataType_MAP:
			return sizeof(connx_Map);
		case connx_DataType_GRAPH:
			return sizeof(connx_Graph);
		default:
			abort();
			return 0;
	}
}

connx_DataType connx_DataType_from_onnx(int32_t type) {
	switch(type) {
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
			return connx_DataType_FLOAT32;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
			return connx_DataType_UINT8;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
			return connx_DataType_INT8;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
			return connx_DataType_UINT16;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
			return connx_DataType_INT16;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
			return connx_DataType_INT32;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
			return connx_DataType_INT64;
		case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
			return connx_DataType_STRING;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
			return connx_DataType_BOOL;
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
			return connx_DataType_FLOAT16;
		case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
			return connx_DataType_FLOAT64;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
			return connx_DataType_UINT32;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
			return connx_DataType_UINT64;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
			return connx_DataType_FLOAT32;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
			return connx_DataType_FLOAT64;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
			return connx_DataType_FLOAT16;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
		default:
			return connx_DataType_VOID;
	}
}

connx_Tensor* connx_Tensor_create(connx_DataType type, uint32_t dimension, ...) {
	uint32_t* lengths = connx_alloc(sizeof(uint32_t) * dimension);

	uint32_t total = 1;
	va_list list;
	va_start(list, dimension);
	for(uint32_t i = 0; i < dimension; i++) {
		lengths[i] = va_arg(list, uint32_t);
		total *= lengths[i];
	}
	va_end(list);

	connx_Tensor* tensor = connx_alloc(sizeof(connx_Tensor) + connx_DataType_size(type) * total);
	tensor->type = connx_DataType_TENSOR;
	tensor->elemType = type;
	tensor->dimension = dimension;
	tensor->lengths = lengths;

	return tensor;
}

connx_Tensor* connx_Tensor_create2(connx_DataType type, uint32_t dimension, uint32_t* lengths) {
	uint32_t* lengths2 = connx_alloc(sizeof(uint32_t) * dimension);

	uint32_t total = 1;
	for(uint32_t i = 0; i < dimension; i++) {
		lengths2[i] = lengths[i];
		total *= lengths[i];
	}

	connx_Tensor* tensor = connx_alloc(sizeof(connx_Tensor) + connx_DataType_size(type) * total);
	tensor->type = connx_DataType_TENSOR;
	tensor->elemType = type;
	tensor->dimension = dimension;
	tensor->lengths = lengths2;

	return tensor;
}

connx_Tensor* connx_Tensor_create_from_onnx(Onnx__TensorProto* onnx) {
	uint32_t dimension = onnx->n_dims;
	uint32_t lengths[dimension];
	uint32_t total = 1;
	for(size_t i = 0; i < onnx->n_dims; i++) {
		lengths[i] = onnx->dims[i];
		total *= lengths[i];
	}

	connx_DataType datatype = connx_DataType_from_onnx(onnx->data_type);
	connx_Tensor* tensor = connx_Tensor_create2(datatype, dimension, lengths);

	switch(onnx->data_type) {
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
			{
				float* array = (float*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_float_data && idx < total; i++) {
					array[idx++] = onnx->float_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 4) {
					array[idx++] = *(float*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
			{
				uint8_t* array = (uint8_t*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_int32_data && idx < total; i++) {
					array[idx++] = onnx->int32_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 1) {
					array[idx++] = *(uint8_t*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
			{
				uint16_t* array = (uint16_t*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_int32_data && idx < total; i++) {
					array[idx++] = onnx->int32_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 2) {
					array[idx++] = *(uint16_t*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
			{
				int8_t* array = (int8_t*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_int32_data && idx < total; i++) {
					array[idx++] = onnx->int32_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 1) {
					array[idx++] = *(int8_t*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
			{
				int16_t* array = (int16_t*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_int32_data && idx < total; i++) {
					array[idx++] = onnx->int32_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 2) {
					array[idx++] = *(int16_t*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
			{
				int32_t* array = (int32_t*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_int32_data && idx < total; i++) {
					array[idx++] = onnx->int32_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 4) {
					array[idx++] = *(int32_t*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
			{
				// TODO: convert uint16 -> float16 -> float32
				uint16_t* array = (uint16_t*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_int32_data && idx < total; i++) {
					array[idx++] = onnx->int32_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 2) {
					array[idx++] = *(uint16_t*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
			{
				bool* array = (bool*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_int32_data && idx < total; i++) {
					array[idx++] = onnx->int32_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i++) {
					array[idx++] = onnx->raw_data.data[i];
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
			{
				char** array = (char**)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_int32_data && idx < total; i++) {
					array[idx++] = (char*)onnx->string_data[i].data;
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; ) {
					array[idx++] = (char*)(onnx->raw_data.data + i);

					while(onnx->raw_data.data[i] != 0) {
						i++;
					}
					i++;
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
			{
				int64_t* array = (int64_t*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_int64_data && idx < total; i++) {
					array[idx++] = onnx->int64_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 8) {
					array[idx++] = *(int64_t*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
			{
				double* array = (double*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_double_data && idx < total; i++) {
					array[idx++] = onnx->double_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 8) {
					array[idx++] = *(double*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
			{
				uint64_t* array = (uint64_t*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_uint64_data && idx < total; i++) {
					array[idx++] = onnx->uint64_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 4) {
					array[idx++] = *(uint32_t*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
			{
				uint64_t* array = (uint64_t*)tensor->base;
				uint32_t idx = 0;

				for(size_t i = 0; i < onnx->n_uint64_data && idx < total; i++) {
					array[idx++] = onnx->uint64_data[i];
				}

				for(size_t i = 0; i < onnx->raw_data.len && idx < total; i += 8) {
					array[idx++] = *(uint64_t*)(onnx->raw_data.data + i);
				}
			}
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
			// TODO: Implement it
		default:
			abort();
			; // Do nothing
	}

	return tensor;
}

connx_Tensor* connx_Tensor_clone(connx_Tensor* tensor) {
	connx_Tensor* tensor2 = connx_Tensor_create2(tensor->elemType, tensor->dimension, tensor->lengths);
	tensor2->name = tensor->name;
	uint32_t total = connx_Tensor_total(tensor);
	memcpy(tensor2->base, tensor->base, total * connx_DataType_size(tensor->elemType));

	return tensor2;
}

bool connx_Tensor_copy(connx_Tensor* tensor, connx_Tensor* dest) {
	if(tensor->elemType != dest->elemType)
		return false;

	uint32_t total1 = connx_Tensor_total(tensor);
	uint32_t total2 = connx_Tensor_total(dest);
	if(total1 != total2)
		return false;

	memcpy(dest->base, tensor->base, total1 * connx_DataType_size(tensor->elemType));
	return true;
}

void connx_Tensor_clean(connx_Tensor* tensor) {
	uint32_t total = connx_Tensor_total(tensor);

	bzero(tensor->base, total * connx_DataType_size(tensor->elemType));
}

void connx_Tensor_delete(connx_Tensor* tensor) {
	if(tensor->elemType == connx_DataType_STRING) {
		uint32_t total =connx_Tensor_total(tensor);
		char** base = (char**)tensor->base;
		for(uint32_t i = 0; i < total; i++) {
			if(base[i] != NULL) {
				connx_free(base[i]);
			}
		}
	}

	connx_free(tensor->lengths);
	connx_free(tensor);
}

void connx_Tensor_dump(connx_Tensor* tensor) {
	char buf[32];
	connx_DataType_toString(tensor->elemType, 32, buf);
	fprintf(stdout, "%s = %s[ ", tensor->name == NULL ? "<noname>" : tensor->name, buf);
	uint32_t total = 1;
	for(uint32_t i = 0; i < tensor->dimension; i++) {
		total *= tensor->lengths[i];

		fprintf(stdout, "%u", tensor->lengths[i]);
		if(i + 1 < tensor->dimension) {
			fprintf(stdout, ", ");
		}
	}
	fprintf(stdout, " ] = {\n\t");

	uint32_t enter = tensor->lengths[tensor->dimension - 1];
	uint32_t enter2 = -1;
	if(tensor->dimension > 1)
		enter2 = tensor->lengths[tensor->dimension - 2] * enter;

	switch(tensor->elemType) {
		case connx_DataType_VOID:
			break;
		case connx_DataType_UINT8:
			{
				uint8_t* array = (uint8_t*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%u", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_UINT16:
			{
				uint16_t* array = (uint16_t*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%u", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_UINT32:
			{
				uint32_t* array = (uint32_t*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%u", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_UINT64:
			{
				uint64_t* array = (uint64_t*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%lu", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_INT8:
			{
				int8_t* array = (int8_t*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%d", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_INT16:
			{
				int16_t* array = (int16_t*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%d", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_INT32:
			{
				int32_t* array = (int32_t*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%d", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_INT64:
			{
				int64_t* array = (int64_t*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%ld", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_FLOAT16:
			{
				uint16_t* array = (uint16_t*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%f", connx_float16_to_float32(array[i]));
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % enter == 0) {
							fprintf(stdout, "\n\t");

							if((i + 1) % enter2 == 0) {
								fprintf(stdout, "\n\t");
							}
						}
					}
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* array = (float*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%f", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % enter == 0) {
							fprintf(stdout, "\n\t");

							if((i + 1) % enter2 == 0) {
								fprintf(stdout, "\n\t");
							}
						}
					}
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* array = (double*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%f", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_BOOL:
			{
				bool* array = (bool*)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%s", array[i] ? "true" : "false");
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_STRING:
			{
				char** array = (char**)tensor->base;
				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "\"%s\"", array[i]);
					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		default:
			fprintf(stdout, "Illegal type: %d", tensor->elemType);
	}

	fprintf(stdout, "\n}\n");
}

void connx_Tensor_dump_compare(connx_Tensor* tensor, connx_Tensor* tensor2, double epsilon) {
#define RED "\033[0;31m"
#define END "\033[0m"

	char buf[32];
	connx_DataType_toString(tensor->elemType, 32, buf);
	fprintf(stdout, "%s = %s[ ", tensor->name == NULL ? "<noname>" : tensor->name, buf);
	uint32_t total = 1;
	for(uint32_t i = 0; i < tensor->dimension; i++) {
		total *= tensor->lengths[i];

		fprintf(stdout, "%u", tensor->lengths[i]);

		if(i >= tensor2->dimension) {
			fprintf(stdout, RED "(%s)" END, "N/A");
		} else if(tensor2->lengths[i] != tensor->lengths[i]) {
			fprintf(stdout, RED "(%u)" END, tensor2->lengths[i]);
		}

		if(i + 1 < tensor->dimension) {
			fprintf(stdout, ", ");
		}
	}
	fprintf(stdout, " ] = {\n\t");

	uint32_t total2 = connx_Tensor_total(tensor2);

	uint32_t enter = tensor->lengths[tensor->dimension - 1];
	uint32_t enter2 = -1;
	if(tensor->dimension > 1)
		enter2 = tensor->lengths[tensor->dimension - 2] * enter;

	switch(tensor->elemType) {
		case connx_DataType_VOID:
			break;
		case connx_DataType_UINT8:
			{
				uint8_t* array = (uint8_t*)tensor->base;
				uint8_t* array2 = (uint8_t*)tensor2->base;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%u", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(array[i] != array2[i]) {
						fprintf(stdout, RED "(%u)" END, array2[i]);
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_UINT16:
			{
				uint16_t* array = (uint16_t*)tensor->base;
				uint16_t* array2 = (uint16_t*)tensor2->base;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%u", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(array[i] != array2[i]) {
						fprintf(stdout, RED "(%u)" END, array2[i]);
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_UINT32:
			{
				uint32_t* array = (uint32_t*)tensor->base;
				uint32_t* array2 = (uint32_t*)tensor2->base;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%u", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(array[i] != array2[i]) {
						fprintf(stdout, RED "(%u)" END, array2[i]);
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_UINT64:
			{
				uint64_t* array = (uint64_t*)tensor->base;
				uint64_t* array2 = (uint64_t*)tensor2->base;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%lu", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(array[i] != array2[i]) {
						fprintf(stdout, RED "(%lu)" END, array2[i]);
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_INT8:
			{
				int8_t* array = (int8_t*)tensor->base;
				int8_t* array2 = (int8_t*)tensor2->base;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%d", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(array[i] != array2[i]) {
						fprintf(stdout, RED "(%d)" END, array2[i]);
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_INT16:
			{
				int16_t* array = (int16_t*)tensor->base;
				int16_t* array2 = (int16_t*)tensor2->base;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%d", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(array[i] != array2[i]) {
						fprintf(stdout, RED "(%d)" END, array2[i]);
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_INT32:
			{
				int32_t* array = (int32_t*)tensor->base;
				int32_t* array2 = (int32_t*)tensor2->base;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%d", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(array[i] != array2[i]) {
						fprintf(stdout, RED "(%d)" END, array2[i]);
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_INT64:
			{
				int64_t* array = (int64_t*)tensor->base;
				int64_t* array2 = (int64_t*)tensor2->base;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%ld", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(array[i] != array2[i]) {
						fprintf(stdout, RED "(%ld)" END, array2[i]);
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_FLOAT16:
			{
				uint16_t* array = (uint16_t*)tensor->base;
				uint16_t* array2 = (uint16_t*)tensor2->base;
				float diff;

				for(uint32_t i = 0; i < total; i++) {
					float v1 = connx_float16_to_float32(array[i]);
					fprintf(stdout, "%f", v1);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(array[i] == array2[i]) {
						// Do nothing
					} else {
						float v2 = connx_float16_to_float32(array2[i]);
						diff = v1 - v2;

						if(diff < -epsilon || diff > epsilon) {
							fprintf(stdout, RED "(%f)" END, v2);
						}
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % enter == 0) {
							fprintf(stdout, "\n\t");

							if((i + 1) % enter2 == 0) {
								fprintf(stdout, "\n\t");
							}
						}
					}
				}
			}
			break;
		case connx_DataType_FLOAT32:
			{
				float* array = (float*)tensor->base;
				float* array2 = (float*)tensor2->base;
				float diff;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%f", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(isnan(array[i]) && isnan(array2[i])) {
						// Do nothing
					} else if(isnan(array[i]) || isnan(array2[i])) {
						fprintf(stdout, RED "(%f)" END, array2[i]);
					} else if(array[i] != array2[i]) {
						diff = array[i] - array2[i];

						if(diff < -epsilon || diff > epsilon) {
							fprintf(stdout, RED "(%f)" END, array2[i]);
						}
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % enter == 0) {
							fprintf(stdout, "\n\t");

							if((i + 1) % enter2 == 0) {
								fprintf(stdout, "\n\t");
							}
						}
					}
				}
			}
			break;
		case connx_DataType_FLOAT64:
			{
				double* array = (double*)tensor->base;
				double* array2 = (double*)tensor2->base;
				double diff;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%f", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(isnan(array[i]) && isnan(array2[i])) {
						// Do nothing
					} else if(isnan(array[i]) || isnan(array2[i])) {
						fprintf(stdout, RED "(%f)" END, array2[i]);
					} else if(array[i] != array2[i]) {
						diff = array[i] - array2[i];

						if(diff < -epsilon || diff > epsilon) {
							fprintf(stdout, RED "(%f)" END, array2[i]);
						}
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_BOOL:
			{
				bool* array = (bool*)tensor->base;
				bool* array2 = (bool*)tensor2->base;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "%s", array[i] ? "true" : "false");

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(array[i] != array2[i]) {
						fprintf(stdout, RED "(%s)" END, array2[i] ? "true" : "false");
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		case connx_DataType_STRING:
			{
				char** array = (char**)tensor->base;
				char** array2 = (char**)tensor2->base;

				for(uint32_t i = 0; i < total; i++) {
					fprintf(stdout, "\"%s\"", array[i]);

					if(i >= total2) {
						fprintf(stdout, RED "(%s)" END, "N/A");
					} else if(strcmp(array[i],array2[i]) != 0) {
						fprintf(stdout, RED "(%s)" END, array2[i]);
					}

					if(i + 1 < total) {
						fprintf(stdout, ", ");

						if((i + 1) % 8 == 0)
							fprintf(stdout, "\n\t");
					}
				}
			}
			break;
		default:
			fprintf(stdout, "Illegal type: %d", tensor->elemType);
	}

	fprintf(stdout, "\n}\n");
}

uint32_t connx_Tensor_total(connx_Tensor* tensor) {
	uint32_t total = 1;
	for(uint32_t i = 0; i < tensor->dimension; i++) {
		total *= tensor->lengths[i];
	}

	return total;
}

bool connx_Tensor_equals(connx_Tensor* tensor, connx_Tensor* tensor2) {
	if(!connx_Tensor_isShapeEquals(tensor, tensor2)) {
		return false;
	}

	uint32_t count = connx_Tensor_total(tensor);

	switch(tensor->elemType) {
		case connx_DataType_UINT8:
		case connx_DataType_UINT16:
		case connx_DataType_UINT32:
		case connx_DataType_UINT64:
		case connx_DataType_INT8:
		case connx_DataType_INT16:
		case connx_DataType_INT32:
		case connx_DataType_INT64:
		case connx_DataType_BOOL:
			return memcmp(tensor->base, tensor2->base, connx_DataType_size(tensor->elemType) * count) == 0;
		case connx_DataType_FLOAT16:
			{
				uint16_t* base = (uint16_t*)tensor->base;
				uint16_t* base2 = (uint16_t*)tensor2->base;

				for(uint32_t i = 0; i < count; i++) {
					if(base[i] == base2[i])
						continue;

					return false;
				}

				return true;
			}
		case connx_DataType_FLOAT32:
			{
				float* base = (float*)tensor->base;
				float* base2 = (float*)tensor2->base;

				for(uint32_t i = 0; i < count; i++) {
					if(base[i] == base2[i] || (isnan(base[i]) && isnan(base2[i])))
						continue;

					return false;
				}

				return true;
			}
		case connx_DataType_FLOAT64:
			{
				double* base = (double*)tensor->base;
				double* base2 = (double*)tensor2->base;

				for(uint32_t i = 0; i < count; i++) {
					if(base[i] == base2[i] || (isnan(base[i]) && isnan(base2[i])))
						continue;

					return false;
				}

				return true;
			}
		case connx_DataType_STRING:
			{
				char** base = (char**)tensor->base;
				char** base2 = (char**)tensor2->base;

				for(uint32_t i = 0; i < count; i++) {
					if(strcmp(base[i], base2[i]) != 0) {
						return false;
					}
				}

				return true;
			}
		default:
			return false;
	}
}

bool connx_Tensor_isNearlyEquals(connx_Tensor* tensor, connx_Tensor* tensor2, double epsilon) {
	if(!connx_Tensor_isShapeEquals(tensor, tensor2)) {
		return false;
	}

	uint32_t count = connx_Tensor_total(tensor);

	switch(tensor->elemType) {
		case connx_DataType_UINT8:
		case connx_DataType_UINT16:
		case connx_DataType_UINT32:
		case connx_DataType_UINT64:
		case connx_DataType_INT8:
		case connx_DataType_INT16:
		case connx_DataType_INT32:
		case connx_DataType_INT64:
		case connx_DataType_BOOL:
			return memcmp(tensor->base, tensor2->base, connx_DataType_size(tensor->elemType) * count) == 0;
		case connx_DataType_FLOAT16:
			{
				uint16_t* base = (uint16_t*)tensor->base;
				uint16_t* base2 = (uint16_t*)tensor2->base;
				float e = epsilon;

				for(uint32_t i = 0; i < count; i++) {
					if(base[i] == base2[i])
						continue;

					float diff = connx_float16_to_float32(base[i]) - connx_float16_to_float32(base2[i]);
					if(diff >= -e && diff <= e)
						continue;

					return false;
				}

				return true;
			}
		case connx_DataType_FLOAT32:
			{
				float* base = (float*)tensor->base;
				float* base2 = (float*)tensor2->base;
				float e = epsilon;

				for(uint32_t i = 0; i < count; i++) {
					if(base[i] == base2[i])
						continue;

					if(isnan(base[i]) && isnan(base2[i]))
						continue;

					float diff = base[i] - base2[i];
					if(diff >= -e && diff <= e)
						continue;

					return false;
				}

				return true;
			}
		case connx_DataType_FLOAT64:
			{
				double* base = (double*)tensor->base;
				double* base2 = (double*)tensor2->base;
				double e = epsilon;

				for(uint32_t i = 0; i < count; i++) {
					if(base[i] == base2[i])
						continue;

					if(isnan(base[i]) && isnan(base2[i]))
						continue;

					double diff = base[i] - base2[i];
					if(diff >= -e && diff <= e)
						continue;

					return false;
				}

				return true;
			}
		case connx_DataType_STRING:
			{
				char** base = (char**)tensor->base;
				char** base2 = (char**)tensor2->base;

				if(epsilon == 0.0) {
					for(uint32_t i = 0; i < count; i++) {
						if(strcmp(base[i], base2[i]) != 0) {
							return false;
						}
					}
				} else {
					int prefix = 0;
					while(epsilon < 1.0) {
						prefix++;
						epsilon *= 10;
					}

					for(uint32_t i = 0; i < count; i++) {
						if(strcmp(base[i], base2[i]) == 0) {
							continue;
						}

						char* pos1 = strchr(base[i], '.');
						char* pos2 = strchr(base2[i], '.');

						if(pos1 != NULL && pos2 != NULL) {
							int prefixLen1 = (int)(pos1 - base[i]);
							int prefixLen2 = (int)(pos2 - base2[i]);
							if(prefixLen1 != prefixLen2) {
								return false;
							}

							int strLen1 = strlen(base[i]);
							int strLen2 = strlen(base2[i]);

							int len = prefixLen1 + prefix;
							len = len < strLen1 ? len : strLen1;
							len = len < strLen2 ? len : strLen2;

							if(memcmp(base[i], base2[i], len) == 0) {
								continue;
							}
						}

						return false;
					}
				}

				return true;
			}
		default:
			return false;
	}
}

bool connx_Tensor_isShapeEquals(connx_Tensor* tensor, connx_Tensor* tensor2) {
	if(tensor->elemType != tensor2->elemType)
		return false;

	if(tensor->dimension != tensor2->dimension)
		return false;

	for(uint32_t i = 0; i < tensor->dimension; i++) {
		if(tensor->lengths[i] != tensor2->lengths[i])
			return false;
	}

	return true;
}

int connx_Tensor_toShapeString(connx_Tensor* tensor, int len, char* buf) {
	int pos = connx_DataType_toString(tensor->elemType, len, buf);
	pos += snprintf(buf, len, "[ ");
	for(uint32_t i = 0; i < tensor->dimension; i++) {
		pos += snprintf(buf + pos, len - pos, "%u", tensor->lengths[i]);
		if(i + 1 < tensor->dimension) {
			pos += snprintf(buf + pos, len - pos, ", ");
		}
	}
	snprintf(buf + pos, len - pos, " ]");

	return pos;
}

connx_Sequence* connx_Sequence_create(connx_DataType type, uint32_t length) {
	connx_Sequence* seq = connx_alloc(sizeof(connx_Sequence) + connx_DataType_size(type) * length);
	seq->type = connx_DataType_SEQUENCE;
	seq->elemType = type;
	seq->length = length;

	return seq;
}

connx_Map* connx_Map_create(connx_DataType keyType, connx_DataType valueType, uint32_t length) {
	connx_Map* map = connx_alloc(sizeof(connx_Map));
	map->type = connx_DataType_SEQUENCE;
	map->keyType = keyType;
	map->valueType = valueType;
	map->length = length;

	return map;
}

connx_Value* connx_Value_create_from_onnx(connx_Type* type) {
	switch(type->value_case) {
		case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
			{
				connx_Type_Tensor* tensor = type->tensor_type;
				uint32_t lengths[tensor->shape->n_dim];
				for(uint32_t i = 0; i < tensor->shape->n_dim; i++) {
					switch(tensor->shape->dim[i]->value_case) {
						case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
							lengths[i] = tensor->shape->dim[i]->dim_value;
							break;
						case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
							lengths[i] = 0;
							break;
						default:
							lengths[i] = 0;
					}
				}

				return (connx_Value*)connx_Tensor_create2(connx_DataType_from_onnx(tensor->elem_type), tensor->shape->n_dim, lengths);
			}
			break;
		case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
			{
				// TODO: Implement it
				// connx_Type_Sequence* seq = valueInfo->type->sequence_type;
				// value = (connx_Value*)connx_Sequence_create(connx_DataType_from_onnx(seq->elem_type), 0);
			}
			break;
		case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
			{
				// TODO: Implement it
				// connx_Type_Map* map = valueInfo->type->map_type;
				// value = (connx_Value*)connx_Map_create(connx_DataType_from_onnx(map->key_type), connx_DataType_from_onnx(map->value_type), 0);
			}
			break;
		default:
			;
	}

	return NULL;
}

connx_Value* connx_Value_clone(connx_Value* value) {
	switch(value->type) {
		case connx_DataType_TENSOR:
			return (connx_Value*)connx_Tensor_clone((connx_Tensor*)value);
		// TODO: Implement seq, map
		default:
			abort();
			return NULL;
	}
}

bool connx_Value_copy(connx_Value* value, connx_Value* dest) {
	switch(value->type) {
		case connx_DataType_TENSOR:
			return connx_Tensor_copy((connx_Tensor*)value, (connx_Tensor*)dest);
		// TODO: Implement seq, map
		default:
			abort();
			return false;
	}
}

void connx_Value_clean(connx_Value* value) {
	switch(value->type) {
		case connx_DataType_TENSOR:
			connx_Tensor_clean((connx_Tensor*)value);
			break;
		// TODO: Implement seq, map
		default:
			abort();
	}
}

void connx_Value_delete(connx_Value* value) {
	switch(value->type) {
		case connx_DataType_TENSOR:
			connx_Tensor_delete((connx_Tensor*)value);
			break;
		// TODO: Implement seq, map
		default:
			;
	}
}

uintptr_t connx_Attribute_create_float(float v) {
	float* p = connx_alloc(sizeof(float));
	*p = v;

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_create_int(int64_t v) {
	int64_t* p = connx_alloc(sizeof(int64_t));
	*p = v;

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_create_string(const char* v) {
	int len = strlen(v) + 1;
	char* p = connx_alloc(len);
	memcpy(p, v, len);

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_create_string_from_onnx(ProtobufCBinaryData* data) {
	int len = data->len + 1;
	char* p = connx_alloc(len);
	memcpy(p, data->data, len - 1);
	p[data->len] = 0;

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_create_floats(uint32_t length, float* v) {
	void* p = connx_alloc(sizeof(uint32_t) + sizeof(float) * length);
	*(uint32_t*)p = length;
	memcpy(p + sizeof(uint32_t), v, sizeof(float) * length);

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_create_ints(uint32_t length, int64_t* v) {
	void* p = connx_alloc(sizeof(uint32_t) + sizeof(int64_t) * length);
	*(uint32_t*)p = length;
	memcpy(p + sizeof(uint32_t), v, sizeof(int64_t) * length);

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_create_strings(uint32_t length, char** v) {
	void* p = connx_alloc(sizeof(uint32_t) + sizeof(char*) * length);
	*(uint32_t*)p = length;
	memcpy(p + sizeof(uint32_t), v, sizeof(char*) * length);

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_create_strings_from_onnx(uint32_t length, ProtobufCBinaryData** v) {
	uint32_t total = 0;
	for(uint32_t i = 0; i < length; i++) {
		total += v[i]->len + 1;
	}

	void* p = connx_alloc(sizeof(uint32_t) + sizeof(char*) * length + total);
	*(uint32_t*)p = length;
	char** array = p + sizeof(uint32_t);
	char* str = p + sizeof(uint32_t) + sizeof(char*) * length;

	for(uint32_t i = 0; i < length; i++) {
		array[i] = str;
		memcpy(str, v[i]->data, v[i]->len);
		str += v[i]->len + 1;
	}

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_clone_float(uintptr_t attr) {
	float* p = connx_alloc(sizeof(float));
	*p = *(float*)attr;

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_clone_int(uintptr_t attr) {
	int64_t* p = connx_alloc(sizeof(int64_t));
	*p = *(int64_t*)attr;

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_clone_string(uintptr_t attr) {
	int len = strlen((char*)attr) + 1;
	char* p = connx_alloc(len);
	memcpy(p, (char*)attr, len);

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_clone_floats(uintptr_t attr) {
	uint32_t length = connx_Attribute_length((void*)attr);
	void* p = connx_alloc(sizeof(uint32_t) + sizeof(float) * length);
	*(uint32_t*)p = length;
	memcpy(p + sizeof(uint32_t), connx_Attribute_base((void*)attr), sizeof(float) * length);

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_clone_ints(uintptr_t attr) {
	uint32_t length = connx_Attribute_length((void*)attr);
	void* p = connx_alloc(sizeof(uint32_t) + sizeof(int64_t) * length);
	*(uint32_t*)p = length;
	memcpy(p + sizeof(uint32_t), connx_Attribute_base((void*)attr), sizeof(int64_t) * length);

	return (uintptr_t)p;
}

uintptr_t connx_Attribute_clone_strings(uintptr_t attr) {
	uint32_t length = connx_Attribute_length((void*)attr);
	void* p = connx_alloc(sizeof(uint32_t) + sizeof(char*) * length);
	*(uint32_t*)p = length;
	memcpy(p + sizeof(uint32_t), connx_Attribute_base((void*)attr), sizeof(char*) * length);

	return (uintptr_t)p;
}

void connx_Attribute_delete(void* attr) {
	connx_free(attr);
}

uint32_t connx_Attribute_length(void* attr) {
	return *(uint32_t*)attr;
}

void* connx_Attribute_base(void* attr) {
	return attr + sizeof(uint32_t);
}

static connx_Tensor* _Graph_getInitializer(connx_Graph* graph, const char* name) {
	for(uint32_t i = 0; i < graph->n_initializer; i++) {
		if(strcmp(graph->initializer[i]->name, name) == 0) {
			return connx_Tensor_create_from_onnx(graph->initializer[i]);
		}
	}

	return NULL;
}

static bool _Graph_hasInput(connx_Graph* graph, const char* name) {
	for(uint32_t i = 0; i < graph->n_input; i++) {
		if(strcmp(graph->input[i]->name, name) == 0)
			return true;
	}

	return false;
}

static bool _Graph_hasOutput(connx_Graph* graph, const char* name) {
	for(uint32_t i = 0; i < graph->n_output; i++) {
		if(strcmp(graph->output[i]->name, name) == 0)
			return true;
	}

	return false;
}

static bool _Graph_hasValueInfo(connx_Graph* graph, const char* name) {
	for(uint32_t i = 0; i < graph->n_value_info; i++) {
		if(strcmp(graph->value_info[i]->name, name) == 0)
			return true;
	}

	return false;
}

static bool _Graph_hasInitializer(connx_Graph* graph, const char* name) {
	for(uint32_t i = 0; i < graph->n_initializer; i++) {
		if(strcmp(graph->initializer[i]->name, name) == 0)
			return true;
	}

	return false;
}

/**
 * Find variable in input, output or value_info
 *
 * @return 1 if the value is in input
 *         2 if the value is in output
 *         3 if the value is in value_info
 *         0 if the value is not in input, output or value_info
 */
static int _Graph_hasValue(connx_Graph* graph, const char* name) {
	if(_Graph_hasInput(graph, name))
		return 1;
	else if(_Graph_hasOutput(graph, name))
		return 2;
	else if(_Graph_hasValueInfo(graph, name))
		return 3;
	else
		return 0;
}

struct ds_Ring {
	void*(*alloc)(size_t);
	void(*free)(void*);

	uint32_t	head;
	uint32_t	tail;
	uint32_t	size;
	uint32_t	count;
	uint8_t		base[0];
};

struct ds_Ring* ds_Ring_create(uint32_t size, uint32_t count, void*(*alloc)(size_t), void(*free)(void*)) {
	struct ds_Ring* ring = calloc(sizeof(struct ds_Ring), (size * count));
	if(ring == NULL)
		return NULL;

	ring->alloc = alloc;
	ring->free = free;

	ring->head = 0;
	ring->tail = 0;
	ring->size = size;
	ring->count = count;

	return ring;
}

void ds_Ring_delete(struct ds_Ring* ring) {
	free(ring);
}

bool ds_Ring_enqueue(struct ds_Ring* ring, void* ptr) {
	uint32_t next = (ring->tail + 1) % ring->count;
	if(next == ring->head)
		return false;

	((uintptr_t*)ring->base)[ring->tail] = (uintptr_t)ptr;
	ring->tail = next;

	return true;
}

bool ds_Ring_dequeue(struct ds_Ring* ring, void* ptr) {
	if(ring->head == ring->tail)
		return false;

	*(uintptr_t*)ptr = ((uintptr_t*)ring->base)[ring->head];
	ring->head = (ring->head + 1) % ring->count;

	return true;
}

struct _ds_ListNode {
	struct _ds_ListNode*	prev;
	struct _ds_ListNode*	next;
	uint8_t					base[0];
};

struct ds_List {
	void*(*alloc)(size_t);
	void(*free)(void*);

	bool(*equals)(void*, void*);

	uint32_t				size;
	uint32_t				count;

	struct _ds_ListNode*	head;
	struct _ds_ListNode*	tail;
};

static bool _ds_List_equals_string(void* v1, void* v2) {
	return strcmp(v1, v2) == 0;
}

static bool _ds_List_equals_ptr(void* v1, void* v2) {
	return v1 == v2;
}

struct ds_List* ds_List_create(uint32_t size, bool(*equals)(void*, void*), void*(*alloc)(size_t), void(*free)(void*)) {
	struct ds_List* list = calloc(sizeof(struct ds_List), 1);
	if(list == NULL)
		return NULL;

	list->alloc = alloc;
	list->free = free;

	if(equals == NULL)
		list->equals = _ds_List_equals_ptr;
	else
		list->equals = equals;

	list->head = NULL;
	list->tail = NULL;
	list->size = size;
	list->count = 0;

	return list;
}

void ds_List_delete(struct ds_List* list) {
	struct _ds_ListNode* node = list->head;
	while(node != NULL) {
		struct _ds_ListNode* next = node->next;
		list->free(node);

		node = next;
	}

	free(list);
}

bool ds_List_add(struct ds_List* list, void* ptr) {
	struct _ds_ListNode* node = list->alloc(sizeof(struct _ds_ListNode) + list->size);
	if(node == NULL)
		return false;

	*(uintptr_t*)node->base = (uintptr_t)ptr;

	if(list->tail == NULL) {
		list->head = list->tail = node;
	} else {
		list->tail->next = node;
		node->prev = list->tail;
		list->tail = node;
	}

	list->count++;

	return true;
}

void* ds_List_get(struct ds_List* list, uint32_t idx) {
	uint32_t i = 0;
	struct _ds_ListNode* node = list->head;
	while(node != NULL) {
		if(i++ == idx) {
			return (void*)*(uintptr_t*)node->base;
		}

		node = node->next;
	}

	return NULL;
}

static void _ds_List_remove(struct ds_List* list, struct _ds_ListNode* node) {
	if(node->prev != NULL)
		node->prev->next = node->next;

	if(node->next != NULL)
		node->next->prev = node->prev;

	if(list->head == node)
		list->head = node->next;

	if(list->tail == node)
		list->tail = node->prev;

	list->free(node);

	list->count--;
}

bool ds_List_remove(struct ds_List* list, void* ptr) {
	struct _ds_ListNode* node = list->head;
	while(node != NULL) {
		if(list->equals((void*)*(uintptr_t*)node->base, ptr)) {
			_ds_List_remove(list, node);

			return true;
		}

		node = node->next;
	}

	return false;
}

bool ds_List_removeAt(struct ds_List* list, uint32_t idx, void* ptr) {
	struct _ds_ListNode* node = list->head;
	while(idx-- != 0) {
		if(node == NULL)
			return false;

		node = node->next;
	}

	if(ptr != NULL)
		*(uintptr_t*)ptr = *(uintptr_t*)node->base;

	_ds_List_remove(list, node);

	return true;
}

bool ds_List_contains(struct ds_List* list, void* ptr) {
	struct _ds_ListNode* node = list->head;
	while(node != NULL) {
		if(list->equals((void*)*(uintptr_t*)node->base, ptr))
			return true;

		node = node->next;
	}

	return false;
}

uint32_t ds_List_clear(struct ds_List* list) {
	uint32_t count = list->count;

	struct _ds_ListNode* node = list->head;
	while(node != NULL) {
		struct _ds_ListNode* next = node->next;
		list->free(node);
		node = next;
	}

	list->head = list->tail = NULL;
	list->count = 0;

	return count;
}

struct ds_ListIterator {
	struct ds_List*			list;
	struct _ds_ListNode*	node;
	struct _ds_ListNode*	prev;
};

struct ds_ListIterator ds_ListIterator_create(struct ds_List* list) {
	struct ds_ListIterator iter;
	iter.list = list;
	iter.node = list->head;
	iter.prev = NULL;

	return iter;
}

bool ds_ListIterator_hasNext(struct ds_ListIterator* iter) {
	return iter->node != NULL;
}

void* ds_ListIterator_next(struct ds_ListIterator* iter) {
	void* ptr = (void*)*(uintptr_t*)iter->node->base;
	iter->prev = iter->node;
	iter->node = iter->node->next;

	return ptr;
}

bool ds_ListIterator_remove(struct ds_ListIterator* iter) {
	if(iter->prev != NULL) {
		struct _ds_ListNode* prev = iter->prev;
		iter->prev = prev->prev;

		_ds_List_remove(iter->list, prev);
		return true;
	} else {
		return false;
	}
}

static uint32_t _Graph_findNodesByOutputs(connx_Graph* graph, connx_Node** nodes, struct ds_List* names) {
	uint32_t idx = 0;
	for(uint32_t i = 0; i < graph->n_node; i++) {
		// Try to match node
		connx_Node* node = graph->node[i];
		char** outputs = node->output;

		// check match and remove from names
		bool isMatch = false;

		for(struct ds_ListIterator iterNames = ds_ListIterator_create(names); ds_ListIterator_hasNext(&iterNames); ) {
			char* name = ds_ListIterator_next(&iterNames);

			for(uint32_t j = 0; j < node->n_output; j++) {
				if(strcmp(outputs[j], name) == 0) {
					ds_ListIterator_remove(&iterNames);
					isMatch = true;
				}
			}
		}

		if(isMatch) {
			nodes[idx++] = node;
		}
	}

	return idx;
}

connx_Path* connx_Path_create() {
	connx_Path* path = connx_alloc(sizeof(connx_Path));
	return path;
}

void connx_Path_delete(connx_Path* path) {
	if(path->outputPaths != NULL)
		connx_free(path->outputPaths);

	if(path->inputPaths != NULL)
		connx_free(path->inputPaths);

	if(path->outputNames != NULL)
		connx_free(path->outputNames);

	if(path->inputNames != NULL)
		connx_free(path->inputNames);

	if(path->nodes != NULL)
		connx_free(path->nodes);

	if(path->operators != NULL)
		connx_free(path->operators);

	connx_free(path);
}

bool connx_Path_hasOutputName(connx_Path* path, char* name) {
	for(uint32_t i = 0; i < path->outputNameCount; i++) {
		if(strcmp(path->outputNames[i], name) == 0) {
			return true;
		}
	}

	return false;
}

void connx_Path_addOutputName(connx_Path* path, char* name) {
	path->outputNameCount++;
	if(path->outputNames != NULL) {
		path->outputNames = connx_realloc(path->outputNames, sizeof(char*) * path->outputNameCount);
	} else {
		path->outputNames = connx_alloc(sizeof(char*) * path->outputNameCount);
	}

	path->outputNames[path->outputNameCount - 1] = name;
}

void connx_Path_addNode(connx_Path* path, connx_Node* node) {
	if(path->outputNames == NULL) {
		path->outputNameCount = node->n_output;
		if(path->outputNames != NULL) {
			path->outputNames = connx_realloc(path->outputNames, sizeof(char*) * path->outputNameCount);
		} else {
			path->outputNames = connx_alloc(sizeof(char*) * path->outputNameCount);
		}
		memcpy(path->outputNames, node->output, sizeof(char*) * path->outputNameCount);
	}

	path->inputNameCount = node->n_input;
	if(path->inputNames != NULL) {
		path->inputNames = connx_realloc(path->inputNames, sizeof(char*) * path->inputNameCount);
	} else {
		path->inputNames = connx_alloc(sizeof(char*) * path->inputNameCount);
	}
	memcpy(path->inputNames, node->input, sizeof(char*) * path->inputNameCount);

	path->count++;
	if(path->nodes != NULL) {
		path->nodes = connx_realloc(path->nodes, sizeof(connx_Node*) * path->count);
	} else {
		path->nodes = connx_alloc(sizeof(connx_Node*) * path->count);
	}
	path->nodes[path->count - 1] = node;
}

void connx_Path_addPath(connx_Path* path, connx_Path* inputPath) {
	path->inputPathCount++;
	if(path->inputPaths != NULL) {
		path->inputPaths = connx_realloc(path->inputPaths, sizeof(connx_Path*) * path->inputPathCount);
	} else {
		path->inputPaths = connx_alloc(sizeof(connx_Path*) * path->inputPathCount);
	}
	path->inputPaths[path->inputPathCount - 1] = inputPath;

	inputPath->outputPathCount++;
	if(inputPath->outputPaths != NULL) {
		inputPath->outputPaths = connx_realloc(inputPath->outputPaths, sizeof(connx_Path*) * inputPath->outputPathCount);
	} else {
		inputPath->outputPaths = connx_alloc(sizeof(connx_Path*) * inputPath->outputPathCount);
	}
	inputPath->outputPaths[inputPath->outputPathCount - 1] = path;
}

#ifdef __linux__
struct PThread {
	pthread_t	pthread;
	void*		context;
};
#endif /* __linux__ */

connx_Thread* connx_Thread_create(struct _connx_Runtime* runtime, connx_Path** paths, uint32_t count) {
	connx_Thread* thread = connx_alloc(sizeof(connx_Thread) + sizeof(struct PThread));

	thread->runtime = runtime;
	thread->paths = connx_alloc(sizeof(connx_Path*) * count);
	thread->pathCount = count;
	memcpy(thread->paths, paths, sizeof(connx_Path*) * count);

	return thread;
}

static uint32_t Stack_init(connx_Runtime* runtime, connx_Node* node, connx_Operator* op, uintptr_t* stack) {
	// set stack count
	uintptr_t count = 1 + node->n_output + node->n_input + op->attributeCount;
	if(op->isOutputVarArgs)
		count++;
	if(op->isInputVarArgs)
		count++;

	uint32_t stackIdx = 0;
	stack[stackIdx++] = count;

	// check output count
	if(op->isOutputVarArgs) {
		if(node->n_output < op->outputCount) {
			connx_exception("Output count too small: name: %s, op: %u, node: %u", node->name, op->outputCount, node->n_output);
			return 0;
		}

		stack[stackIdx++] = (uintptr_t)op->outputCount;
	} else {
		if(node->n_output != op->outputCount) {
			connx_exception("Output count mismatch: name: %s, op: %u, node: %u", node->name, op->outputCount, node->n_output);
			return 0;
		}
	}

	// missing variables
	uint32_t missing[op->outputCount];
	uint32_t missingCount = 0;

	// set output
	for(uint32_t i = 0; i < op->outputCount; i++) {
		connx_Value* value = connx_Runtime_getVariable(runtime, node->output[i]);
		stack[stackIdx++] = (uintptr_t)value;

		if(value == NULL) {
			missing[missingCount++] = i;
		}
	}

	// check input count
	if(op->isInputVarArgs) {
		if(node->n_input < op->inputCount) {
			connx_exception("Input count too small: name: %s, op: %u, node: %u", node->name, op->inputCount, node->n_input);
			return 0;
		}

		stack[stackIdx++] = (uintptr_t)op->inputCount;
	} else {
		if(node->n_input != op->inputCount) {
			connx_exception("Input count mismatch: name: %s, op: %u, node: %u", node->name, op->inputCount, node->n_input);
			return 0;
		}
	}

	// set input
	for(uint32_t i = 0; i < op->inputCount; i++) {
		connx_Value* value = connx_Runtime_getVariable(runtime, node->input[i]);
		if(value == NULL) {
			connx_exception("Cannot find input variable: %s for operator %s(%s)", node->input[i], node->name, node->op_type);
			return 0;
		}

		stack[stackIdx++] = (uintptr_t)value;
	}

	// attribute
	for(uint32_t i = 0; i < op->attributeCount; i++) {
		// find attribute
		connx_Attribute* attr = NULL;
		for(uint32_t j = 0; j < node->n_attribute; j++) {
			connx_Attribute* a = node->attribute[j];
			if(strcmp(a->name, op->attributeNames[i]) == 0) {
				attr = a;
				break;
			}
		}

		if(attr != NULL) {
			// set attribute
			switch(attr->type) {
				case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT:
					stack[stackIdx++] = connx_Attribute_create_float(attr->f);
					break;
				case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT:
					stack[stackIdx++] = connx_Attribute_create_int(attr->i);
					break;
				case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING:
					stack[stackIdx++] = connx_Attribute_create_string_from_onnx(&attr->s);
					break;
				case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS:
					stack[stackIdx++] = connx_Attribute_create_floats(attr->n_floats, attr->floats);
					break;
				case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS:
					stack[stackIdx++] = connx_Attribute_create_ints(attr->n_ints, attr->ints);
					break;
				case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS:
					stack[stackIdx++] = connx_Attribute_create_strings_from_onnx(attr->n_strings, &attr->strings);
					break;
				default:
					// TODO: implement it
					abort();
			}
		} else {
			// set default attribute
			switch(op->attributes[i]) {
				case connx_DataType_FLOAT32:
					stack[stackIdx++] = connx_Attribute_clone_float(op->attributeValues[i]);
					break;
				case connx_DataType_INT64:
					stack[stackIdx++] = connx_Attribute_clone_int(op->attributeValues[i]);
					break;
				case connx_DataType_STRING:
					stack[stackIdx++] = connx_Attribute_clone_string(op->attributeValues[i]);
					break;
				case connx_DataType_FLOAT32_ARRAY:
					stack[stackIdx++] = connx_Attribute_clone_floats(op->attributeValues[i]);
					break;
				case connx_DataType_INT64_ARRAY:
					stack[stackIdx++] = connx_Attribute_clone_ints(op->attributeValues[i]);
					break;
				case connx_DataType_STRING_ARRAY:
					stack[stackIdx++] = connx_Attribute_clone_strings(op->attributeValues[i]);
					break;
				default:
					// TODO: implement it
					abort();
			}
		}
	}

	// resolve
	if(!op->resolve(stack)) {
		connx_exception("Validation failed node: %s(%s)", node->name, op->name);
		return 0;
	}

	// create missing variables
	if(missingCount > 0) {
		uint32_t oldVariableCount = runtime->variableCount;
		runtime->variableCount += missingCount;

		// TODO: change to hashmap
		runtime->variables = connx_realloc(runtime->variables, sizeof(connx_Value*) * runtime->variableCount);
		runtime->initializers = connx_realloc(runtime->initializers, sizeof(connx_Value*) * runtime->variableCount);

		for(uint32_t i = 0; i < missingCount; i++) {
			connx_Value* value = (connx_Value*)stack[1 + missing[i]];
			value->name = node->output[missing[i]];

			runtime->initializers[oldVariableCount + i] = value;
			runtime->variables[oldVariableCount + i] = connx_Value_clone(value);
		}
	}

	assert(count == stackIdx);

	return stackIdx;
}

bool connx_Thread_init(connx_Thread* thread) {
	// get stack count
	uint32_t stackCount = 0;
	for(uint32_t i = 0; i < thread->pathCount; i++) {
		connx_Path* path = thread->paths[i];

		for(uint32_t j = 0; j < path->count; j++) {
			connx_Operator* op = path->operators[j];

			stackCount += 1 + op->outputCount + op->inputCount + op->attributeCount;

			if(op->isOutputVarArgs)
				stackCount++;

			if(op->isInputVarArgs)
				stackCount++;
		}
	}

	// make stack
	thread->stackCount = stackCount;
	thread->stack = connx_alloc(sizeof(uintptr_t) * thread->stackCount);

	// init stack
	uint32_t stackIdx = 0;
	uintptr_t* stack = thread->stack;
	for(uint32_t i = 0; i < thread->pathCount; i++) {
		connx_Path* path = thread->paths[i];

		for(uint32_t j = 0; j < path->count; j++) {
			connx_Node* node = path->nodes[j];
			connx_Operator* op = path->operators[j];

			uint32_t count = Stack_init(thread->runtime, node, op, stack + stackIdx);
			if(count == 0)
				return false;

			stackIdx += count;
			assert(stackIdx <= stackCount);
		}
	}

	return true;
}

void connx_Thread_delete(connx_Thread* thread) {
#ifdef __linux__
	// Do nothing
#endif /* __linux__ */

	uint32_t stackIdx = 0;
	uintptr_t* stack = thread->stack;
	for(uint32_t i = 0; i < thread->pathCount; i++) {
		connx_Path* path = thread->paths[i];

		for(uint32_t j = 0; j < path->count; j++) {
			connx_Operator* op = path->operators[j];

			stackIdx++;	// count

			if(op->isOutputVarArgs)
				stackIdx++;	// output count

			stackIdx += op->outputCount;	// outputs

			if(op->isInputVarArgs)
				stackIdx++;	// output count

			stackIdx += op->inputCount;	// inputs

			for(uint32_t j = 0; j < op->attributeCount; j++)
				connx_Attribute_delete((void*)stack[stackIdx++]);
		}

		connx_free(stack);
		connx_free(thread->paths);
	}

	connx_free(thread);
}

bool connx_Thread_start(connx_Thread* thread) {
#ifdef __linux__
	struct PThread* priv = (void*)thread->priv;
	int ret = pthread_create(&priv->pthread, NULL, (void*)connx_Thread_run, thread);

	switch(ret) {
		case 0:
			return true;
		default:
			connx_exception("Cannot start POSIX thread: %d", ret);
			return false;
	}
#endif /* __linux__ */
}

void* connx_Thread_run(connx_Thread* thread) {
	uint32_t stackIdx = 0;
	uintptr_t* stack = thread->stack;
	for(uint32_t i = 0; i < thread->pathCount; i++) {
		connx_Path* path = thread->paths[i];
		connx_Operator** ops = path->operators;

		for(uint32_t j = 0; j < path->count; j++) {
			uintptr_t count = stack[stackIdx];
			if(!ops[j]->exec(stack + stackIdx))
				return NULL;

			stackIdx += count;
		}
	}

	return NULL;
}

bool connx_Thread_join(connx_Thread* thread) {
#ifdef __linux__
	struct PThread* priv = (void*)thread->priv;
	int ret = pthread_join(priv->pthread, NULL);

	switch(ret) {
		case 0:
			return true;
		default:
			connx_exception("Cannot join POSIX thread: %d", ret);
			return false;
	}
#endif /* __linux__ */
}

connx_Runtime* connx_Runtime_create(connx_Model* model) {
	connx_Runtime* runtime = connx_alloc(sizeof(connx_Runtime));
	if(runtime == NULL) {
		connx_exception("Out of memory");
		return NULL;
	}

	runtime->model = model;

	// Dependency calculation and scheduling
	connx_Graph* graph = model->graph;

	// outputs
	uint32_t outputIdx = 0;
	connx_ValueInfo* outputs[graph->n_output];
	for(uint32_t i = 0; i < graph->n_output; i++) {
		connx_ValueInfo* output = graph->output[i];

		for(uint32_t j = 0; j < graph->n_node; j++) {
			connx_Node* node = graph->node[j];
			for(uint32_t k = 0; k < node->n_input; k++) {
				if(strcmp(node->input[k], output->name) == 0) {
					goto next_output;
				}
			}
		}

		outputs[outputIdx++] = output;
next_output:
		;
	}

	runtime->outputCount = outputIdx;
	runtime->outputs = connx_alloc(sizeof(connx_ValueInfo*) * outputIdx);
	memcpy(runtime->outputs, outputs, sizeof(connx_ValueInfo*) * outputIdx);

	// inputs
	uint32_t inputIdx = 0;
	connx_ValueInfo* inputs[graph->n_input];
	for(uint32_t i = 0; i < graph->n_input; i++) {
		connx_ValueInfo* input = graph->input[i];

		for(uint32_t j = 0; j < graph->n_initializer; j++) {
			if(strcmp(graph->initializer[j]->name, input->name) == 0) {
				goto next_input;
			}
		}

		inputs[inputIdx++] = input;
next_input:
		;
	}

	runtime->inputCount = inputIdx;
	runtime->inputs = connx_alloc(sizeof(connx_ValueInfo*) * inputIdx);
	memcpy(runtime->inputs, inputs, sizeof(connx_ValueInfo*) * inputIdx);

	// orphant initializers(exists in initializer but not in value_info)
	uint32_t orphant_count = 0;
	for(uint32_t i = 0; i < graph->n_initializer; i++) {
		if(_Graph_hasValue(graph, graph->initializer[i]->name) == 0) {
			orphant_count++;
		}
	}

	// variables
	runtime->variableCount = graph->n_input + graph->n_output + graph->n_value_info + orphant_count;
	runtime->variables = connx_alloc(sizeof(connx_Value*) * runtime->variableCount);
	runtime->initializers = connx_alloc(sizeof(connx_Value*) * runtime->variableCount);

	uint32_t variableIdx = 0;
	for(uint32_t i = 0; i < graph->n_input; i++) {
		connx_ValueInfo* valueInfo = graph->input[i];
		connx_Tensor* tensor = _Graph_getInitializer(graph, valueInfo->name);
		if(tensor != NULL) {
			tensor->name = valueInfo->name;
			runtime->initializers[variableIdx] = (connx_Value*)connx_Tensor_clone(tensor);
			runtime->variables[variableIdx++] = (connx_Value*)tensor;
		} else {
			connx_Value* value = connx_Value_create_from_onnx(valueInfo->type);
			value->name = valueInfo->name;
			runtime->variables[variableIdx++] = value;
		}
	}

	for(uint32_t i = 0; i < graph->n_output; i++) {
		connx_ValueInfo* valueInfo = graph->output[i];
		connx_Tensor* tensor = _Graph_getInitializer(graph, valueInfo->name);
		if(tensor != NULL) {
			tensor->name = valueInfo->name;
			runtime->initializers[variableIdx] = (connx_Value*)connx_Tensor_clone(tensor);
			runtime->variables[variableIdx++] = (connx_Value*)tensor;
		} else {
			connx_Value* value = connx_Value_create_from_onnx(valueInfo->type);
			value->name = valueInfo->name;
			runtime->variables[variableIdx++] = value;
		}
	}

	for(uint32_t i = 0; i < graph->n_value_info; i++) {
		connx_ValueInfo* valueInfo = graph->value_info[i];
		connx_Tensor* tensor = _Graph_getInitializer(graph, valueInfo->name);
		if(tensor != NULL) {
			tensor->name = valueInfo->name;
			runtime->initializers[variableIdx] = (connx_Value*)connx_Tensor_clone(tensor);
			runtime->variables[variableIdx++] = (connx_Value*)tensor;
		} else {
			connx_Value* value = connx_Value_create_from_onnx(valueInfo->type);
			value->name = valueInfo->name;
			runtime->variables[variableIdx++] = value;
		}
	}

	if(orphant_count > 0) {
		for(uint32_t i = 0; i < graph->n_initializer; i++) {
			if(_Graph_hasValue(graph, graph->initializer[i]->name) == 0) {
				connx_Tensor* tensor = connx_Tensor_create_from_onnx(graph->initializer[i]);
				tensor->name = graph->initializer[i]->name;

				runtime->initializers[variableIdx] = (connx_Value*)connx_Tensor_clone(tensor);
				runtime->variables[variableIdx++] = (connx_Value*)tensor;
			}
		}
	}

	// Schedule and make paths
	if(!connx_Runtime_schedule(runtime)) {
		return false;
	}

	// threads
	runtime->threadCount = runtime->pathCount - 1;	// except inputPath
	runtime->threads = connx_alloc(sizeof(connx_Thread*) * runtime->pathCount);
	for(uint32_t i = 0; i < runtime->threadCount; i++) {
		connx_Path* paths[] = { runtime->paths[i + 1] };
		runtime->threads[i] = connx_Thread_create(runtime, paths, 1);
		connx_Thread_init(runtime->threads[i]);
	}

	return runtime;
}

void connx_Runtime_delete(connx_Runtime* runtime) {
	for(uint32_t i = 0; i < runtime->threadCount; i++) {
		connx_Thread_delete(runtime->threads[i]);
	}

	connx_free(runtime->threads);

	for(uint32_t i = 0; i < runtime->pathCount; i++) {
		connx_Path_delete(runtime->paths[i]);
	}

	connx_free(runtime->paths);

	for(uint32_t i = 0; i < runtime->variableCount; i++) {
		if(runtime->initializers[i] == NULL)
			continue;

		connx_Value_delete(runtime->initializers[i]);
	}

	connx_free(runtime->initializers);

	for(uint32_t i = 0; i < runtime->variableCount; i++) {
		if(runtime->variables[i] == NULL)
			continue;

		connx_Value_delete(runtime->variables[i]);
	}

	connx_free(runtime->variables);

	connx_free(runtime->outputs);
	connx_free(runtime->inputs);
	connx_free(runtime);
}

static uint32_t _Paths_find(struct ds_List* list, connx_Path** paths, struct ds_List* names) {
	uint32_t idx = 0;

	for(struct ds_ListIterator iterPath = ds_ListIterator_create(list); ds_ListIterator_hasNext(&iterPath); ) {
		connx_Path* path = ds_ListIterator_next(&iterPath);
		char** outputNames = path->outputNames;

		// check match and remove from names
		bool isMatch = false;

		for(struct ds_ListIterator iterNames = ds_ListIterator_create(names); ds_ListIterator_hasNext(&iterNames); ) {
			char* name = ds_ListIterator_next(&iterNames);

			for(uint32_t j = 0; j < path->outputNameCount; j++) {
				if(strcmp(outputNames[j], name) == 0) {
					ds_ListIterator_remove(&iterNames);
					isMatch = true;
				}
			}
		}

		if(isMatch) {
			paths[idx++] = path;
		}
	}

	return idx;
}

/**
 * Algorithm description
 *
 * Graph.inputPath is virtual input path
 * Graph.outputPath is virtual output path
 * unresolved is a queue
 * resolved is a queue
 * push(unresolved, outputPath)
 * push(resolved, inputPath)
 * while(len(unresolved) > 0)
 *     path = peek(unresolved)
 *     if(path.inputNameCount == 0)
 *         pop(unresolved)
 *         continue
 *
 *     nodes, paths = findMatching(path)
 *
 *     if(not found)
 *         exception
 *     
 *     if(len(nodes) == 1 and len(paths) == 0)
 *         extend path
 *     else 
 *         for node in nodes
 *             push(unresolved, newPath(node))
 *
 *         add new paths and old paths to path
 *         pop(unresolved)
 *         push(resolved, path)
 */
bool connx_Runtime_schedule(connx_Runtime* runtime) {
	connx_Graph* graph = runtime->model->graph;

	bool ret = true;

	struct ds_List* unresolved = ds_List_create(sizeof(connx_Path*), NULL, connx_alloc, connx_free);
	struct ds_List* resolved = ds_List_create(sizeof(connx_Path*), NULL, connx_alloc, connx_free);
	struct ds_List* names = ds_List_create(sizeof(char*), _ds_List_equals_string, connx_alloc, connx_free);

	// Make InputPath
	connx_Path* inputPath = connx_Path_create();	// contents will be filled lazily

	// Make OutputPath
	connx_Path* outputPath = connx_Path_create();
	outputPath->inputNameCount = graph->n_output;
	outputPath->inputNames = connx_alloc(sizeof(char*) * outputPath->inputNameCount);
	for(uint32_t i = 0; i < outputPath->inputNameCount; i++)
		outputPath->inputNames[i] = graph->output[i]->name;
	ds_List_add(unresolved, outputPath);

	// Resolve unresolved paths
	while(unresolved->count > 0) {
		connx_Path* path = ds_List_get(unresolved, 0);
		if(path->inputNameCount == 0) {
			ds_List_removeAt(unresolved, 0, NULL);
			continue;
		}

		uint32_t count = path->inputNameCount;
		connx_Node* nodes[count];
		for(uint32_t i = 0; i < count; i++)
			ds_List_add(names, path->inputNames[i]);

		// Find dependent nodes
		uint32_t node_count = _Graph_findNodesByOutputs(graph, nodes, names);

		// Find dependent paths
		connx_Path* paths[count - node_count];
		uint32_t unresolved_path_count = _Paths_find(unresolved, paths, names);
		uint32_t resolved_path_count = _Paths_find(resolved, paths + unresolved_path_count, names);

		// Remove dependency by value_info and initializer
		for(uint32_t i = 0; i < graph->n_value_info; i++) {
			char* name = graph->value_info[i]->name;
			ds_List_remove(names, name);
		}

		for(uint32_t i = 0; i < graph->n_initializer; i++) {
			char* name = graph->initializer[i]->name;
			ds_List_remove(names, name);
		}

		// Check inputs
		bool isInputPathMatch = false;
		for(struct ds_ListIterator iter = ds_ListIterator_create(names); ds_ListIterator_hasNext(&iter); ) {
			char* name = ds_ListIterator_next(&iter);
			for(uint32_t i = 0; i < graph->n_input; i++) {
				if(strcmp(name, graph->input[i]->name) == 0) {
					if(!connx_Path_hasOutputName(inputPath, name))
						connx_Path_addOutputName(inputPath, name);

					ds_ListIterator_remove(&iter);

					isInputPathMatch = true;
				}
			}
		}

		if(isInputPathMatch) {
			paths[resolved_path_count++] = inputPath;
		}

		// Check remain dependency
		uint32_t remain = names->count;
		if(remain > 0) {
			switch(remain) {
				case 1:
					connx_exception("Cannot find dependency %s", ds_List_get(names, 0));
					break;
				case 2:
					connx_exception("Cannot find dependency %s, %s", ds_List_get(names, 0), ds_List_get(names, 1));
					break;
				case 3:
					connx_exception("Cannot find dependency %s, %s, %s", ds_List_get(names, 0), ds_List_get(names, 1), ds_List_get(names, 2));
					break;
				default:
					connx_exception("Cannot find dependency %s, %s, %s and %u more", ds_List_get(names, 0), ds_List_get(names, 1), ds_List_get(names, 2), remain - 3);
			}

			ret = false;
			goto done;
		}

		if(node_count == 1 && unresolved_path_count == 0 && resolved_path_count == 0) {
			connx_Path_addNode(path, nodes[0]);
		} else {
			for(uint32_t i = 0; i < node_count; i++) {
				connx_Path* path2 = connx_Path_create();
				connx_Path_addNode(path2, nodes[i]);
				ds_List_add(unresolved, path2);

				connx_Path_addPath(path, path2);
			}

			for(uint32_t i = 0; i < unresolved_path_count; i++) {
				connx_Path_addPath(path, paths[i]);
			}

			for(uint32_t i = 0; i < resolved_path_count; i++) {
				connx_Path_addPath(path, paths[unresolved_path_count + i]);
			}

			ds_List_removeAt(unresolved, 0, NULL);
			ds_List_add(resolved, path);
		}
	}

	// set runtime
	runtime->inputPath = inputPath;
	runtime->outputPath = outputPath;
	runtime->pathCount = resolved->count + 1;	// + inputPath
	runtime->paths = connx_alloc(sizeof(connx_Path*) * runtime->pathCount);

	// reorder path from back to front (reuse unresolved as a queue)
	assert(unresolved->count == 0);
	ds_List_add(unresolved, runtime->outputPath);
	int32_t idx = resolved->count;
	while(unresolved->count > 0) {
		assert(idx >= 0);

		connx_Path* path;
		ds_List_removeAt(unresolved, 0, &path);
		runtime->paths[idx--] = path;

		// reorder nodes in path
		for(uint32_t i = 0; i < path->count / 2; i++) {
			connx_Node* node = path->nodes[i];
			path->nodes[i] = path->nodes[path->count - i - 1];
			path->nodes[path->count - i - 1] = node;
		}

		// add input path to unresolved
		for(uint32_t i = 0; i < path->inputPathCount; i++) {
			connx_Path* inputPath = path->inputPaths[i];
			bool isDup = false;
			for(uint32_t j = idx + 1; j < resolved->count; j++) {
				if(runtime->paths[j] == inputPath) {
					isDup = true;
					break;
				}
			}

			if(!isDup)
				ds_List_add(unresolved, inputPath);
		}
	}

	assert(idx == -1);

	// resolve operator
	for(uint32_t i = 0; i < runtime->pathCount; i++) {
		connx_Path* path = runtime->paths[i];
		path->operators = connx_alloc(sizeof(connx_Operator*) * path->count);

		for(uint32_t j = 0; j < path->count; j++) {
			path->operators[j] = connx_Operator_get(path->nodes[j]->op_type);
			if(path->operators[j] == NULL) {
				connx_exception("Cannot find operator: %s", path->nodes[j]->op_type);
				goto done;
			}
		}
	}

done:
	// clean up
	ds_List_delete(names);
	ds_List_delete(resolved);
	ds_List_delete(unresolved);

	/*
	printf("pathCount = %u\n", runtime->pathCount);
	for(uint32_t i = 0; i < runtime->pathCount; i++) {
		connx_Path* path = runtime->paths[i];

		printf("path[%u] inputs=", i);
		for(uint32_t j = 0; j < path->inputNameCount; j++) {
			printf("%s ", path->inputNames[j]);
		}

		printf("outputs=");
		for(uint32_t j = 0; j < path->outputNameCount; j++) {
			printf("%s ", path->outputNames[j]);
		}
		printf("\n");

		for(uint32_t j = 0; j < path->count; j++) {
			connx_Node* node = path->nodes[j];
			printf("\tNode[%u]=%s inputs=", j, node->name);

			for(uint32_t k = 0; k < node->n_input; k++) {
				printf("%s ", node->input[k]);
			}

			printf("outputs=");
			for(uint32_t k = 0; k < node->n_output; k++) {
				printf("%s ", node->output[k]);
			}

			printf("op=%s:%s\n", node->name, node->op_type);
		}
	}
	*/

	return ret;
}

bool connx_Runtime_setVariable(connx_Runtime* runtime, connx_Value* value) {
	for(uint32_t i = 0; i < runtime->variableCount; i++) {
		if(strcmp(runtime->variables[i]->name, value->name) == 0) {
			connx_Value* v = (connx_Value*)runtime->variables[i];
			if(v->type != value->type) {
				char buf1[32];
				char buf2[32];
				connx_DataType_toString(v->type, 32, buf1);
				connx_DataType_toString(value->type, 32, buf2);
				connx_exception("Datatype is not matching: %s vs %s", buf1, buf2);
				return false;
			}

			switch(v->type) {
				case connx_DataType_TENSOR:
					{
						connx_Tensor* tensor1 = (connx_Tensor*)v;
						connx_Tensor* tensor2 = (connx_Tensor*)value;

						if(!connx_Tensor_isShapeEquals(tensor1, tensor2)) {
							char buf1[16];
							char buf2[16];
							connx_Tensor_toShapeString(tensor1, 16, buf1);
							connx_Tensor_toShapeString(tensor2, 16, buf2);
							connx_exception("Tensor shape is not matching %s vs %s", buf1, buf2);
							return false;
						}

						uint32_t total = connx_Tensor_total(tensor1);
						memcpy(tensor1->base, tensor2->base, connx_DataType_size(tensor1->elemType) * total);
					}
					break;
				default:
					; // TODO: seq and map
			}

			return true;
		}
	}

	return false;
}

connx_Value* connx_Runtime_getVariable(connx_Runtime* runtime, const char* name) {
	for(uint32_t i = 0; i < runtime->variableCount; i++) {
		if(strcmp(runtime->variables[i]->name, name) == 0) {
			return runtime->variables[i];
		}
	}

	return NULL;
}

connx_Value* connx_Runtime_run(connx_Runtime* runtime, uint32_t inputCount, connx_Value** inputs) {
	// initailize variables
	for(uint32_t i = 0; i < runtime->variableCount; i++) {
		if(runtime->initializers[i] != NULL) {
			connx_Value_copy(runtime->initializers[i], runtime->variables[i]);
		} else {
			connx_Value_clean(runtime->variables[i]);
		}
	}

	// set inputs
	for(uint32_t i = 0; i < inputCount; i++) {
		for(uint32_t j = 0; j < runtime->inputCount; j++) {
			if(strcmp(runtime->inputs[j]->name, inputs[i]->name) == 0) {
				if(!connx_Runtime_setVariable(runtime, inputs[i])) {
					return NULL;
				}
				goto found;
			}
		}

		connx_exception("There is no input variable for: %s", inputs[i]->name);
		return NULL;
found:
		;
	}

	for(uint32_t i = 0; i < runtime->threadCount; i++) {	// except input thread
		connx_Thread_run(runtime->threads[i]);
	}

	if(runtime->outputCount == 1) {
		return connx_Runtime_getVariable(runtime, runtime->outputs[0]->name);
	} else {
		return NULL;
	}

//	if(runtime->isDirty) {
//		for(uint32_t i = 0; i < runtime->variableCount; i++) {
//			if(runtime->initializers[i] != NULL) {
//				connx_Value_copy(runtime->initializers[i], runtime->variables[i]);
//			} else {
//				connx_Value_clean(runtime->variables[i]);
//			}
//		}
//
//		runtime->isDirty = false;
//	} else {
//		runtime->isDirty = true;
//	}
//
//	for(uint32_t i = 0; i < inputCount; i++) {
//		for(uint32_t j = 0; j < runtime->inputCount; j++) {
//			if(strcmp(runtime->inputs[j]->name, inputs[i]->name) == 0) {
//				if(!connx_Runtime_setVariable(runtime, inputs[i])) {
//					return NULL;
//				}
//				goto found;
//			}
//		}
//		connx_exception("There is no input variable for: %s", inputs[i]->name);
//		return NULL;
//found:
//		;
//	}
//
//	uintptr_t* stack = runtime->stack;
//	for(uint32_t i = runtime->dependencyCount; i > 0; i--) {
//		connx_Operator* op = runtime->operators[i - 1];
//
//		if(!op->exec(stack)) {
//			return NULL;
//		}
//
//		/*
//		printf("* dump: %s(%s)\n", runtime->dependencies[i - 1]->name, op->name);
//		uint32_t len = op->outputCount + op->inputCount;
//		for(uint32_t i = 0; i < len; i++) {
//			connx_Tensor_dump((void*)stack[i + 1]);
//		}
//		*/
//
//		uintptr_t count = stack[0];
//		stack += count;
//	}
//
//	if(runtime->outputCount == 1) {
//		return connx_Runtime_getVariable(runtime, runtime->outputs[0]->name);
//	} else {
//		return NULL;
//	}
//
}

void connx_Operator_add(const char* name, 
		uint32_t outputCount, uint32_t inputCount, uint32_t attributeCount,
		bool (*resolve)(uintptr_t* stack), bool (*exec)(uintptr_t* stack), ...) {
	connx_Operator* op = &(connx_operators[connx_operator_count++]);

	int len = strlen(name) + 1;
	op->name = connx_alloc(len);
	memcpy(op->name, name, len);

	if(!!(outputCount & CONNX_VARARGS)) {
		op->isOutputVarArgs = true;
		outputCount ^= CONNX_VARARGS;
	}

	op->outputCount = outputCount;
	op->outputs = connx_alloc(sizeof(connx_DataType) * outputCount);

	if(!!(inputCount & CONNX_VARARGS)) {
		op->isInputVarArgs = true;
		inputCount ^= CONNX_VARARGS;
	}

	op->inputCount = inputCount;
	op->inputs = connx_alloc(sizeof(connx_DataType) * inputCount);

	op->attributeCount = attributeCount;
	op->attributeNames = connx_alloc(sizeof(char*) * attributeCount);
	op->attributes = connx_alloc(sizeof(connx_DataType) * attributeCount);
	op->attributeValues = connx_alloc(sizeof(uintptr_t*) * attributeCount);

	op->resolve = resolve;
	op->exec = exec;

	va_list list;
	va_start(list, exec);

	for(uint32_t i = 0; i < outputCount; i++) {
		op->outputs[i] = va_arg(list, connx_DataType);
	}

	for(uint32_t i = 0; i < inputCount; i++) {
		op->inputs[i] = va_arg(list, connx_DataType);
	}

	for(uint32_t i = 0; i < attributeCount; i++) {
		op->attributeNames[i] = va_arg(list, char*);
		op->attributes[i] = va_arg(list, connx_DataType);
		switch(op->attributes[i]) {
			case connx_DataType_STRING:
				{
					char* v = va_arg(list, char*);
					int len = strlen(v) + 1;
					op->attributeValues[i] = (uintptr_t)connx_alloc(len);
					memcpy((void*)op->attributeValues[i], v, len);
				}
				break;
			case connx_DataType_INT64:
				{
					int32_t v = va_arg(list, int32_t);
					op->attributeValues[i] = connx_Attribute_create_int(v);
				}
				break;
			case connx_DataType_INT64_ARRAY:
				{
					uint32_t count = va_arg(list, uint32_t);
					int64_t* array = va_arg(list, int64_t*);
					op->attributeValues[i] = connx_Attribute_create_ints(count, array);
				}
				break;
			case connx_DataType_FLOAT32:
				{
					float v = va_arg(list, double);
					op->attributeValues[i] = connx_Attribute_create_float(v);
				}
				break;
			case connx_DataType_FLOAT32_ARRAY:
				{
					uint32_t count = va_arg(list, uint32_t);
					float* array = va_arg(list, float*);
					op->attributeValues[i] = connx_Attribute_create_floats(count, array);
				}
				break;
			case connx_DataType_GRAPH:
				{
					void* graph = va_arg(list, void*);
					op->attributeValues[i] = (void*)graph;
				}
				break;
			default:
				// TODO: implement it
				{
					char buf[32];
					connx_DataType_toString(op->attributes[i], 32, buf);
					fprintf(stderr, "attribute datatype %s is not supported yet.\n", buf);
					abort();
				}
		}
	}

	va_end(list);
}

connx_Operator* connx_Operator_get(const char* name) {
	for(uint32_t i = 0; i < connx_operator_count; i++) {
		connx_Operator* op = &(connx_operators[i]);
		if(strcmp(op->name, name) == 0)
			return op;
	}

	return NULL;
}

static int depth = 0;

static void tab() {
	for(int i = 0; i < depth; i++)
		fprintf(stdout, "\t");
}

void connx_Operator_dump() {
	char buf[128];

	for(uint32_t i = 0; i < connx_operator_count; i++) {
		connx_Operator* op = &(connx_operators[i]);
		tab(); fprintf(stdout, "Operator %s\n", op->name);
		depth++;
		for(uint32_t j = 0; j < op->outputCount; j++) {
			connx_DataType_toString(op->outputs[j], 128, buf);
			tab(); fprintf(stdout, "output[%u] = %s\n", j, buf);
		}

		for(uint32_t j = 0; j < op->inputCount; j++) {
			connx_DataType_toString(op->inputs[j], 128, buf);
			tab(); fprintf(stdout, "input[%u] = %s\n", j, buf);
		}

		for(uint32_t j = 0; j < op->attributeCount; j++) {
			tab(); fprintf(stdout, "attribute[%u] ", j);
			depth++;
			if(op->attributes[j] == connx_DataType_STRING) {
				connx_DataType_toString(op->attributes[j], 128, buf);
				fprintf(stdout, "%s : %s = %s\n", 
						op->attributeNames[j], buf, (char*)op->attributeValues[j]);
			} else if((op->attributes[j] | connx_DataType_ARRAY) > 0) {
				connx_DataType_toString(op->attributes[j], 128, buf);
				fprintf(stdout, "%s : %s = %u\n", 
						op->attributeNames[j], buf, *(uint32_t*)op->attributeValues[j]);
			} else {
				connx_DataType_toString(op->attributes[j], 128, buf);
				fprintf(stdout, "%s : %s = %p\n", 
						op->attributeNames[j], buf, (void*)op->attributeValues[j]);
			}
			depth--;
		}
		depth--;
	}
}

// Ref: https://stackoverflow.com/questions/3026441/float32-to-float16/3026505
// Ref: https://gist.github.com/martin-kallman/5049614
// Ref: https://tool.oschina.net/uploads/apidocs/ogre3d/api/html/OgreBitwise_8h_source.html
uint16_t connx_float32_to_float16(float v) {
	uint32_t i = *(uint32_t*)&v;

	int32_t s = (i >> 16) & 0x00008000;
	int32_t e = ((i >> 23) & 0x000000ff) - (127 - 15);
	int32_t m = i & 0x007fffff;

	if(e <= 0) {
		if(e < -10) {
			return 0;
		}
		m = (m | 0x00800000) >> (1 - e);

		return s | (m >> 13);
	} else if(e == 0xff - (127 - 15)) {
		if(m == 0) {	// Inf
			return s | 0x7c00;
		} else {		// NaN
			m >>= 13;
			return s | 0x7c00 | m | (m == 0);
		}
	} else {
		if(e > 30) {	// Overflow
			return s | 0x7c00;
		} else {
			return s | (e << 10) | (m >> 13);
		}
	}
}

float connx_float16_to_float32(uint16_t v) {
	int32_t s = (v >> 15) & 0x00000001;
	int32_t e = (v >> 10) & 0x0000001f;
	int32_t m = v & 0x000003ff;

	uint32_t r;
	if(e == 0) {
		if(m == 0) {	// plus or minus zero
			r = s << 31;
			return *(float*)&r;
		} else {
			while(!(m & 0x00000400)) {
				m <<= 1;
				e -= 1;
			}

			e += 1;
			m &= ~0x00000400;
		}
	} else if(e == 31) {
		if(m == 0) {	// Inf
			r = (s << 31) | 0x7f800000;
			return *(float*)&r;
		} else {		// NaN
			r = (s << 31) | 0x7f800000 | (m << 13);
			return *(float*)&r;
		}
	}

	e += 127 - 15;
	m <<= 13;

	r = (s << 31) | (e << 23) | m;
	return *(float*)&r;
}
