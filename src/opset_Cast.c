#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <connx/connx.h>

static bool Cast_resolve(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];
	int64_t* to = (void*)stack[3];

	// Create output if null
	if(output == NULL) {
		connx_DataType type;

		switch(*to) {
			case 1: // FLOAT = 1
				type = connx_DataType_FLOAT32;
				break;	
			case 2: // UINT8 = 2
				type = connx_DataType_UINT8;
				break;	
			case 3: // INT8 = 3
				type = connx_DataType_INT8;
				break;	
			case 4: // UINT16 = 4
				type = connx_DataType_UINT16;
				break;	
			case 5: // INT16 = 5
				type = connx_DataType_INT16;
				break;	
			case 6: // INT32 = 6
				type = connx_DataType_INT32;
				break;	
			case 7: // INT64 = 7
				type = connx_DataType_INT64;
				break;	
			case 8: // STRING = 8
				type = connx_DataType_STRING;
				break;	
			case 9: // BOOL = 9
				type = connx_DataType_BOOL;
				break;	
			case 10: // FLOAT16 = 10
				type = connx_DataType_FLOAT16;
				break;	
			case 11: // DOUBLE = 11
				type = connx_DataType_FLOAT64;
				break;	
			case 12: // UINT32 = 12
				type = connx_DataType_UINT32;
				break;	
			case 13: // UINT64 = 13
				type = connx_DataType_UINT64;
				break;	
			case 14: // COMPLEX64 = 14
			case 15: // COMPLEX128 = 15
			case 16: // BFLOAT16 = 16
			default:
				connx_exception("Not supported type: %u\n", *to);
				return false;
		}

		output = connx_Tensor_create2(type, input->dimension, input->lengths);
		connx_Stack_update(1, output);
	}

	if(output->dimension != input->dimension) {
		connx_exception("input and output's dimension is differ");
		return false;
	}

	for(uint32_t i = 0; i < output->dimension; i++) {
		if(input->lengths[i] != output->lengths[i]) {
			connx_exception("input and output length is differ: dimension: %u, input length: %u, output length: %u", output->dimension, input->lengths[i], output->lengths[i]);
			return false;
		}
	}

	switch(*to) {
		case 1: // FLOAT = 1
			if(output->elemType != connx_DataType_FLOAT32) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 2: // UINT8 = 2
			if(output->elemType != connx_DataType_UINT8) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 3: // INT8 = 3
			if(output->elemType != connx_DataType_INT8) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 4: // UINT16 = 4
			if(output->elemType != connx_DataType_UINT16) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 5: // INT16 = 5
			if(output->elemType != connx_DataType_INT16) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 6: // INT32 = 6
			if(output->elemType != connx_DataType_INT32) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 7: // INT64 = 7
			if(output->elemType != connx_DataType_INT64) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 8: // STRING = 8
			if(output->elemType != connx_DataType_STRING) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 9: // BOOL = 9
			if(output->elemType != connx_DataType_BOOL) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 10: // FLOAT16 = 10
			if(output->elemType != connx_DataType_FLOAT16) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 11: // DOUBLE = 11
			if(output->elemType != connx_DataType_FLOAT64) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 12: // UINT32 = 12
			if(output->elemType != connx_DataType_UINT32) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 13: // UINT64 = 13
			if(output->elemType != connx_DataType_UINT64) {
				char buf[32];
				connx_DataType_toString(output->elemType, 32, buf);
				connx_exception("output's element type and to attribute is differ: %s vs %lu", buf, *to);
				return false;
			}
			break;	
		case 14: // COMPLEX64 = 14
		case 15: // COMPLEX128 = 15
		case 16: // BFLOAT16 = 16
		default:
			connx_exception("Not supported type: %u\n", *to);
			return false;
	}

	return true;
}

static bool copy(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);

	uint32_t output_size = connx_DataType_size(output->elemType);
	uint32_t input_size = connx_DataType_size(input->elemType);

	uint32_t total = output_total < input_total ? output_total : input_total;
	uint32_t size = output_size < input_size ? output_size : input_size;

	memcpy(output->base, input->base, total * size);

	return true;
}

#define CAST(input_type, output_type) 											\
	uint32_t output_total = connx_Tensor_total(output);							\
	uint32_t input_total = connx_Tensor_total(input);							\
	uint32_t total = output_total < input_total ? output_total : input_total;	\
																				\
	output_type* output_base = (output_type*)output->base;						\
	input_type* input_base = (input_type*)input->base;							\
	for(uint32_t i = 0; i < total; i++) {										\
		*output_base++ = *input_base++;											\
	}																			\
																				\
	return true;

// uint8
static bool uint8_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint8_t, uint16_t)
}

static bool uint8_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint8_t, uint32_t)
}

static bool uint8_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint8_t, uint64_t)
}

static bool uint8_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint8_t, int8_t)
}

static bool uint8_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint8_t, int16_t)
}

static bool uint8_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint8_t, int32_t)
}

static bool uint8_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint8_t, int64_t)
}

static bool uint8_to_float16(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	uint8_t* input_base = (uint8_t*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16((float)*input_base++);
	}

	return true;
}

static bool uint8_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint8_t, float)
}

static bool uint8_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint8_t, double)
}

static bool uint8_to_bool(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	uint8_t* input_base = (uint8_t*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = !!*input_base++;
	}

	return true;
}

static bool uint8_to_string(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint8_t* input_base = (uint8_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%u", *input_base++);
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// uint16
static bool uint16_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint16_t, uint8_t)
}

static bool uint16_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint16_t, uint32_t)
}

static bool uint16_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint16_t, uint64_t)
}

static bool uint16_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint16_t, int8_t)
}

static bool uint16_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint16_t, int16_t)
}

static bool uint16_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint16_t, int32_t)
}

static bool uint16_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint16_t, int64_t)
}

static bool uint16_to_float16(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	uint16_t* input_base = (uint16_t*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16((float)*input_base++);
	}

	return true;
}

static bool uint16_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint16_t, float)
}

static bool uint16_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint16_t, double)
}

static bool uint16_to_bool(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	uint16_t* input_base = (uint16_t*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = !!*input_base++;
	}

	return true;
}

static bool uint16_to_string(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint16_t* input_base = (uint16_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%u", *input_base++);
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// uint32
static bool uint32_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint32_t, uint8_t)
}

static bool uint32_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint32_t, uint16_t)
}

static bool uint32_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint32_t, uint64_t)
}

static bool uint32_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint32_t, int8_t)
}

static bool uint32_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint32_t, int16_t)
}

static bool uint32_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint32_t, int32_t)
}

static bool uint32_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint32_t, int64_t)
}

static bool uint32_to_float16(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	uint32_t* input_base = (uint32_t*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16((float)*input_base++);
	}

	return true;
}

static bool uint32_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint32_t, float)
}

static bool uint32_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint32_t, double)
}

static bool uint32_to_bool(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	uint32_t* input_base = (uint32_t*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = !!*input_base++;
	}

	return true;
}

static bool uint32_to_string(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint32_t* input_base = (uint32_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%u", *input_base++);
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// uint16
static bool uint64_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint64_t, uint8_t)
}

static bool uint64_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint64_t, uint16_t)
}

static bool uint64_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint64_t, uint32_t)
}

static bool uint64_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint64_t, int8_t)
}

static bool uint64_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint64_t, int16_t)
}

static bool uint64_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint64_t, int32_t)
}

static bool uint64_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint64_t, int64_t)
}

static bool uint64_to_float16(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	uint64_t* input_base = (uint64_t*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16((float)*input_base++);
	}

	return true;
}

static bool uint64_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint64_t, float)
}

static bool uint64_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST(uint64_t, double)
}

static bool uint64_to_bool(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	uint64_t* input_base = (uint64_t*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = !!*input_base++;
	}

	return true;
}

static bool uint64_to_string(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint64_t* input_base = (uint64_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%lu", *input_base++);
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// int8
static bool int8_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST(int8_t, uint16_t)
}

static bool int8_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int8_t, uint32_t)
}

static bool int8_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int8_t, uint64_t)
}

static bool int8_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST(int8_t, int8_t)
}

static bool int8_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST(int8_t, int16_t)
}

static bool int8_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int8_t, int32_t)
}

static bool int8_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int8_t, int64_t)
}

static bool int8_to_float16(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	int8_t* input_base = (int8_t*)input->base;
	for(int32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16((float)*input_base++);
	}

	return true;
}

static bool int8_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int8_t, float)
}

static bool int8_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int8_t, double)
}

static bool int8_to_bool(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	int8_t* input_base = (int8_t*)input->base;
	for(int32_t i = 0; i < total; i++) {
		*output_base++ = !!*input_base++;
	}

	return true;
}

static bool int8_to_string(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	int8_t* input_base = (int8_t*)input->base;
	char buf[32];
	for(int32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%u", *input_base++);
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// int16
static bool int16_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST(int16_t, uint8_t)
}

static bool int16_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int16_t, uint32_t)
}

static bool int16_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int16_t, uint64_t)
}

static bool int16_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST(int16_t, int8_t)
}

static bool int16_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST(int16_t, int16_t)
}

static bool int16_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int16_t, int32_t)
}

static bool int16_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int16_t, int64_t)
}

static bool int16_to_float16(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	int16_t* input_base = (int16_t*)input->base;
	for(int32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16((float)*input_base++);
	}

	return true;
}

static bool int16_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int16_t, float)
}

static bool int16_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int16_t, double)
}

static bool int16_to_bool(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	int16_t* input_base = (int16_t*)input->base;
	for(int32_t i = 0; i < total; i++) {
		*output_base++ = !!*input_base++;
	}

	return true;
}

static bool int16_to_string(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	int16_t* input_base = (int16_t*)input->base;
	char buf[32];
	for(int32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%u", *input_base++);
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// int32
static bool int32_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST(int32_t, uint8_t)
}

static bool int32_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST(int32_t, uint16_t)
}

static bool int32_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int32_t, uint64_t)
}

static bool int32_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST(int32_t, int8_t)
}

static bool int32_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST(int32_t, int16_t)
}

static bool int32_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int32_t, int32_t)
}

static bool int32_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int32_t, int64_t)
}

static bool int32_to_float16(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	int32_t* input_base = (int32_t*)input->base;
	for(int32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16((float)*input_base++);
	}

	return true;
}

static bool int32_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int32_t, float)
}

static bool int32_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int32_t, double)
}

static bool int32_to_bool(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	int32_t* input_base = (int32_t*)input->base;
	for(int32_t i = 0; i < total; i++) {
		*output_base++ = !!*input_base++;
	}

	return true;
}

static bool int32_to_string(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	int32_t* input_base = (int32_t*)input->base;
	char buf[32];
	for(int32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%u", *input_base++);
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// int16
static bool int64_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST(int64_t, uint8_t)
}

static bool int64_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST(int64_t, uint16_t)
}

static bool int64_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int64_t, uint32_t)
}

static bool int64_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST(int64_t, int8_t)
}

static bool int64_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST(int64_t, int16_t)
}

static bool int64_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int64_t, int32_t)
}

static bool int64_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int64_t, int64_t)
}

static bool int64_to_float16(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	int64_t* input_base = (int64_t*)input->base;
	for(int32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16((float)*input_base++);
	}

	return true;
}

static bool int64_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST(int64_t, float)
}

static bool int64_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST(int64_t, double)
}

static bool int64_to_bool(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	int64_t* input_base = (int64_t*)input->base;
	for(int32_t i = 0; i < total; i++) {
		*output_base++ = !!*input_base++;
	}

	return true;
}

static bool int64_to_string(connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	int64_t* input_base = (int64_t*)input->base;
	char buf[32];
	for(int32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%lu", *input_base++);
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// float16
#define CAST_FLOAT16(type)														\
	uint32_t output_total = connx_Tensor_total(output);							\
	uint32_t input_total = connx_Tensor_total(input);							\
	uint32_t total = output_total < input_total ? output_total : input_total;	\
																				\
	type* output_base = (type*)output->base;									\
	uint16_t* input_base = (uint16_t*)input->base;								\
	for(uint32_t i = 0; i < total; i++) {										\
		*output_base++ = connx_float16_to_float32(*input_base++);				\
	}																			\
																				\
	return true;

static bool float16_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST_FLOAT16(uint8_t)
}

static bool float16_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST_FLOAT16(uint16_t)
}

static bool float16_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST_FLOAT16(uint32_t)
}

static bool float16_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST_FLOAT16(uint64_t)
}

static bool float16_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST_FLOAT16(int8_t)
}

static bool float16_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST_FLOAT16(int16_t)
}

static bool float16_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST_FLOAT16(int32_t)
}

static bool float16_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST_FLOAT16(int64_t)
}

static bool float16_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST_FLOAT16(float)
}

static bool float16_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST_FLOAT16(double)
}

static bool float16_to_bool(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	uint16_t* input_base = (uint16_t*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = connx_float16_to_float32(*input_base++) != 0;
	}

	return true;
}

static bool float16_to_string(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint16_t* input_base = (uint16_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%f", connx_float16_to_float32(*input_base++));
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// float32
static bool float32_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST(float, uint8_t)
}

static bool float32_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST(float, uint16_t)
}

static bool float32_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST(float, uint32_t)
}

static bool float32_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST(float, uint64_t)
}

static bool float32_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST(float, int8_t)
}

static bool float32_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST(float, int16_t)
}

static bool float32_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST(float, int32_t)
}

static bool float32_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST(float, int64_t)
}

static bool float32_to_float16(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	float* input_base = (float*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16((float)*input_base++);
	}

	return true;
}

static bool float32_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST(float, double)
}

static bool float32_to_bool(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	float* input_base = (float*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = *input_base++ != 0;
	}

	return true;
}

static bool float32_to_string(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	float* input_base = (float*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%.9f", *input_base++);
		char* buf2 = connx_alloc(len + 1);
		memcpy(buf2, buf, len);
		buf2[len] = '\0';
		*output_base++ = buf2;
	}

	return true;
}

// float64
static bool float64_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST(double, uint8_t)
}

static bool float64_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST(double, uint16_t)
}

static bool float64_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST(double, uint32_t)
}

static bool float64_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST(double, uint64_t)
}

static bool float64_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST(double, int8_t)
}

static bool float64_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST(double, int16_t)
}

static bool float64_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST(double, int32_t)
}

static bool float64_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST(double, int64_t)
}

static bool float64_to_float16(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	double* input_base = (double*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16((float)*input_base++);
	}

	return true;
}

static bool float64_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST(double, float)
}

static bool float64_to_bool(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint8_t* output_base = (uint8_t*)output->base;
	double* input_base = (double*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = *input_base++ != 0;
	}

	return true;
}

static bool float64_to_string(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	double* input_base = (double*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%.17f", *input_base++);
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// bool
#define CAST_BOOL(type)															\
	uint32_t output_total = connx_Tensor_total(output);							\
	uint32_t input_total = connx_Tensor_total(input);							\
	uint32_t total = output_total < input_total ? output_total : input_total;	\
																				\
	type* output_base = (type*)output->base;									\
	uint16_t* input_base = (uint16_t*)input->base;								\
	for(uint32_t i = 0; i < total; i++) {										\
		*output_base++ = *input_base++ ? 1 : 0;									\
	}																			\
																				\
	return true;

static bool bool_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST_BOOL(uint8_t)
}

static bool bool_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST_BOOL(uint16_t)
}

static bool bool_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST_BOOL(uint32_t)
}

static bool bool_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST_BOOL(uint64_t)
}

static bool bool_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST_BOOL(int8_t)
}

static bool bool_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST_BOOL(int16_t)
}

static bool bool_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST_BOOL(int32_t)
}

static bool bool_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST_BOOL(int64_t)
}

static bool bool_to_float16(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	double* input_base = (double*)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16(*input_base++ ? 1 : 0);
	}

	return true;
}

static bool bool_to_float32(connx_Tensor* input, connx_Tensor* output) {
	CAST_BOOL(float)
}

static bool bool_to_float64(connx_Tensor* input, connx_Tensor* output) {
	CAST_BOOL(double)
}

static bool bool_to_string(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	double* input_base = (double*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%s", *input_base++ ? "true" : "false");
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// bool
#define CAST_STRING(type, func)															\
	uint32_t output_total = connx_Tensor_total(output);							\
	uint32_t input_total = connx_Tensor_total(input);							\
	uint32_t total = output_total < input_total ? output_total : input_total;	\
																				\
	type* output_base = (type*)output->base;									\
	char** input_base = (char**)input->base;								\
	for(uint32_t i = 0; i < total; i++) {										\
		*output_base++ = func(*input_base++, NULL, 10);									\
	}																			\
																				\
	return true;

static bool string_to_uint8(connx_Tensor* input, connx_Tensor* output) {
	CAST_STRING(uint8_t, strtoul)
}

static bool string_to_uint16(connx_Tensor* input, connx_Tensor* output) {
	CAST_STRING(uint16_t, strtoul)
}

static bool string_to_uint32(connx_Tensor* input, connx_Tensor* output) {
	CAST_STRING(uint32_t, strtoul)
}

static bool string_to_uint64(connx_Tensor* input, connx_Tensor* output) {
	CAST_STRING(uint64_t, strtoull)
}

static bool string_to_int8(connx_Tensor* input, connx_Tensor* output) {
	CAST_STRING(int8_t, strtol)
}

static bool string_to_int16(connx_Tensor* input, connx_Tensor* output) {
	CAST_STRING(int16_t, strtol)
}

static bool string_to_int32(connx_Tensor* input, connx_Tensor* output) {
	CAST_STRING(int32_t, strtol)
}

static bool string_to_int64(connx_Tensor* input, connx_Tensor* output) {
	CAST_STRING(int64_t, strtoll)
}

static bool string_to_float16(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	uint16_t* output_base = (uint16_t*)output->base;
	char** input_base = (char**)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = connx_float32_to_float16(strtof(*input_base++, NULL));
	}

	return true;
}

static bool string_to_float32(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	float* output_base = (float*)output->base;
	char** input_base = (char**)input->base;
	for(uint32_t i = 0; i < total; i++) {
		char* str = *input_base++;
		switch(str[1]) {
			case 'a':	// NaN
				if(strncmp("NaN", str, 3) == 0) {
					*output_base++ = NAN;
				} else {
					*output_base++ = NAN;
				}
				break;
			case 'N':	// INF
				if(strncmp("INF", str, 3) == 0) {
					*output_base++ = INFINITY;
				} else {
					*output_base++ = NAN;
				}
				break;
			case 'I':	// ±INF
				if(strncmp("+INF", str, 4) == 0) {
					*output_base++ = INFINITY;
				} else if(strncmp("-INF", str, 4) == 0) {
					*output_base++ = -INFINITY;
				} else {
					*output_base++ = NAN;
				}
				break;
			default:
				*output_base++ = strtof(str, NULL);
		}
	}

	return true;
}

static bool string_to_float64(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	float* output_base = (float*)output->base;
	char** input_base = (char**)input->base;
	for(uint32_t i = 0; i < total; i++) {
		*output_base++ = strtod(*input_base++, NULL);
	}

	return true;
}

static bool string_to_bool(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	double* input_base = (double*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%s", *input_base++ ? "true" : "false");
		char* buf2 = connx_alloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

static bool string_to_string(__attribute__((unused)) connx_Tensor* input, __attribute__((unused)) connx_Tensor* output) {
	abort();
	return false;
}

static bool Cast_exec(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];
	__attribute__((unused)) int64_t* to = (void*)stack[3];

	switch(input->elemType) {
		case connx_DataType_UINT8:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return copy(input, output);
				case connx_DataType_UINT16:
					return uint8_to_uint16(input, output);
				case connx_DataType_UINT32:
					return uint8_to_uint32(input, output);
				case connx_DataType_UINT64:
					return uint8_to_uint64(input, output);
				case connx_DataType_INT8:
					return uint8_to_int8(input, output);
				case connx_DataType_INT16:
					return uint8_to_int16(input, output);
				case connx_DataType_INT32:
					return uint8_to_int32(input, output);
				case connx_DataType_INT64:
					return uint8_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return uint8_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return uint8_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return uint8_to_float64(input, output);
				case connx_DataType_BOOL:
					return uint8_to_bool(input, output);
				case connx_DataType_STRING:
					return uint8_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_UINT16:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return uint16_to_uint8(input, output);
				case connx_DataType_UINT16:
					return copy(input, output);
				case connx_DataType_UINT32:
					return uint16_to_uint32(input, output);
				case connx_DataType_UINT64:
					return uint16_to_uint64(input, output);
				case connx_DataType_INT8:
					return uint16_to_int8(input, output);
				case connx_DataType_INT16:
					return uint16_to_int16(input, output);
				case connx_DataType_INT32:
					return uint16_to_int32(input, output);
				case connx_DataType_INT64:
					return uint16_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return uint16_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return uint16_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return uint16_to_float64(input, output);
				case connx_DataType_BOOL:
					return uint16_to_bool(input, output);
				case connx_DataType_STRING:
					return uint16_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_UINT32:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return uint32_to_uint8(input, output);
				case connx_DataType_UINT16:
					return uint32_to_uint16(input, output);
				case connx_DataType_UINT32:
					return copy(input, output);
				case connx_DataType_UINT64:
					return uint32_to_uint64(input, output);
				case connx_DataType_INT8:
					return uint32_to_int8(input, output);
				case connx_DataType_INT16:
					return uint32_to_int16(input, output);
				case connx_DataType_INT32:
					return uint32_to_int32(input, output);
				case connx_DataType_INT64:
					return uint32_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return uint32_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return uint32_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return uint32_to_float64(input, output);
				case connx_DataType_BOOL:
					return uint32_to_bool(input, output);
				case connx_DataType_STRING:
					return uint32_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_UINT64:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return uint64_to_uint8(input, output);
				case connx_DataType_UINT16:
					return uint64_to_uint16(input, output);
				case connx_DataType_UINT32:
					return uint64_to_uint32(input, output);
				case connx_DataType_UINT64:
					return copy(input, output);
				case connx_DataType_INT8:
					return uint64_to_int8(input, output);
				case connx_DataType_INT16:
					return uint64_to_int16(input, output);
				case connx_DataType_INT32:
					return uint64_to_int32(input, output);
				case connx_DataType_INT64:
					return uint64_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return uint64_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return uint64_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return uint64_to_float64(input, output);
				case connx_DataType_BOOL:
					return uint64_to_bool(input, output);
				case connx_DataType_STRING:
					return uint64_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_INT8:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return copy(input, output);
				case connx_DataType_UINT16:
					return int8_to_uint16(input, output);
				case connx_DataType_UINT32:
					return int8_to_uint32(input, output);
				case connx_DataType_UINT64:
					return int8_to_uint64(input, output);
				case connx_DataType_INT8:
					return int8_to_int8(input, output);
				case connx_DataType_INT16:
					return int8_to_int16(input, output);
				case connx_DataType_INT32:
					return int8_to_int32(input, output);
				case connx_DataType_INT64:
					return int8_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return int8_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return int8_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return int8_to_float64(input, output);
				case connx_DataType_BOOL:
					return int8_to_bool(input, output);
				case connx_DataType_STRING:
					return int8_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_INT16:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return int16_to_uint8(input, output);
				case connx_DataType_UINT16:
					return copy(input, output);
				case connx_DataType_UINT32:
					return int16_to_uint32(input, output);
				case connx_DataType_UINT64:
					return int16_to_uint64(input, output);
				case connx_DataType_INT8:
					return int16_to_int8(input, output);
				case connx_DataType_INT16:
					return int16_to_int16(input, output);
				case connx_DataType_INT32:
					return int16_to_int32(input, output);
				case connx_DataType_INT64:
					return int16_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return int16_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return int16_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return int16_to_float64(input, output);
				case connx_DataType_BOOL:
					return int16_to_bool(input, output);
				case connx_DataType_STRING:
					return int16_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_INT32:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return int32_to_uint8(input, output);
				case connx_DataType_UINT16:
					return int32_to_uint16(input, output);
				case connx_DataType_UINT32:
					return copy(input, output);
				case connx_DataType_UINT64:
					return int32_to_uint64(input, output);
				case connx_DataType_INT8:
					return int32_to_int8(input, output);
				case connx_DataType_INT16:
					return int32_to_int16(input, output);
				case connx_DataType_INT32:
					return int32_to_int32(input, output);
				case connx_DataType_INT64:
					return int32_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return int32_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return int32_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return int32_to_float64(input, output);
				case connx_DataType_BOOL:
					return int32_to_bool(input, output);
				case connx_DataType_STRING:
					return int32_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_INT64:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return int64_to_uint8(input, output);
				case connx_DataType_UINT16:
					return int64_to_uint16(input, output);
				case connx_DataType_UINT32:
					return int64_to_uint32(input, output);
				case connx_DataType_UINT64:
					return copy(input, output);
				case connx_DataType_INT8:
					return int64_to_int8(input, output);
				case connx_DataType_INT16:
					return int64_to_int16(input, output);
				case connx_DataType_INT32:
					return int64_to_int32(input, output);
				case connx_DataType_INT64:
					return int64_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return int64_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return int64_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return int64_to_float64(input, output);
				case connx_DataType_BOOL:
					return int64_to_bool(input, output);
				case connx_DataType_STRING:
					return int64_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_FLOAT16:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return float16_to_uint8(input, output);
				case connx_DataType_UINT16:
					return float16_to_uint16(input, output);
				case connx_DataType_UINT32:
					return float16_to_uint32(input, output);
				case connx_DataType_UINT64:
					return float16_to_uint64(input, output);
				case connx_DataType_INT8:
					return float16_to_int8(input, output);
				case connx_DataType_INT16:
					return float16_to_int16(input, output);
				case connx_DataType_INT32:
					return float16_to_int32(input, output);
				case connx_DataType_INT64:
					return float16_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return copy(input, output);
				case connx_DataType_FLOAT32:
					return float16_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return float16_to_float64(input, output);
				case connx_DataType_BOOL:
					return float16_to_bool(input, output);
				case connx_DataType_STRING:
					return float16_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_FLOAT32:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return float32_to_uint8(input, output);
				case connx_DataType_UINT16:
					return float32_to_uint16(input, output);
				case connx_DataType_UINT32:
					return float32_to_uint32(input, output);
				case connx_DataType_UINT64:
					return float32_to_uint64(input, output);
				case connx_DataType_INT8:
					return float32_to_int8(input, output);
				case connx_DataType_INT16:
					return float32_to_int16(input, output);
				case connx_DataType_INT32:
					return float32_to_int32(input, output);
				case connx_DataType_INT64:
					return float32_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return float32_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return copy(input, output);
				case connx_DataType_FLOAT64:
					return float32_to_float64(input, output);
				case connx_DataType_BOOL:
					return float32_to_bool(input, output);
				case connx_DataType_STRING:
					return float32_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_FLOAT64:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return float64_to_uint8(input, output);
				case connx_DataType_UINT16:
					return float64_to_uint16(input, output);
				case connx_DataType_UINT32:
					return float64_to_uint32(input, output);
				case connx_DataType_UINT64:
					return float64_to_uint64(input, output);
				case connx_DataType_INT8:
					return float64_to_int8(input, output);
				case connx_DataType_INT16:
					return float64_to_int16(input, output);
				case connx_DataType_INT32:
					return float64_to_int32(input, output);
				case connx_DataType_INT64:
					return float64_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return float64_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return float64_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return copy(input, output);
				case connx_DataType_BOOL:
					return float64_to_bool(input, output);
				case connx_DataType_STRING:
					return float64_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_BOOL:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return bool_to_uint8(input, output);
				case connx_DataType_UINT16:
					return bool_to_uint16(input, output);
				case connx_DataType_UINT32:
					return bool_to_uint32(input, output);
				case connx_DataType_UINT64:
					return bool_to_uint64(input, output);
				case connx_DataType_INT8:
					return bool_to_int8(input, output);
				case connx_DataType_INT16:
					return bool_to_int16(input, output);
				case connx_DataType_INT32:
					return bool_to_int32(input, output);
				case connx_DataType_INT64:
					return bool_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return bool_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return bool_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return bool_to_float64(input, output);
				case connx_DataType_BOOL:
					return copy(input, output);
				case connx_DataType_STRING:
					return bool_to_string(input, output);
				default:
					return false;
			}
			break;
		case connx_DataType_STRING:
			switch(output->elemType) {
				case connx_DataType_UINT8:
					return string_to_uint8(input, output);
				case connx_DataType_UINT16:
					return string_to_uint16(input, output);
				case connx_DataType_UINT32:
					return string_to_uint32(input, output);
				case connx_DataType_UINT64:
					return string_to_uint64(input, output);
				case connx_DataType_INT8:
					return string_to_int8(input, output);
				case connx_DataType_INT16:
					return string_to_int16(input, output);
				case connx_DataType_INT32:
					return string_to_int32(input, output);
				case connx_DataType_INT64:
					return string_to_int64(input, output);
				case connx_DataType_FLOAT16:
					return string_to_float16(input, output);
				case connx_DataType_FLOAT32:
					return string_to_float32(input, output);
				case connx_DataType_FLOAT64:
					return string_to_float64(input, output);
				case connx_DataType_BOOL:
					return string_to_bool(input, output);
				case connx_DataType_STRING:
					return string_to_string(input, output);
				default:
					return false;
			}
			break;
		default:
			return false;
	}

	return true;
}

bool connx_opset_Cast_init() {
	connx_Operator_add("Cast", 1, 1, 1, Cast_resolve, Cast_exec,
		connx_DataType_TENSOR_NUMBER | connx_DataType_BOOL | connx_DataType_STRING,	// output
		connx_DataType_TENSOR_NUMBER | connx_DataType_BOOL | connx_DataType_STRING,	// input
		"to", connx_DataType_INT64, 0);

	return true;
}
