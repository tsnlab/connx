#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <connx/connx.h>

static bool Cast_resolve(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];
	int64_t* to = (void*)stack[3];

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

static bool copy(connx_Tensor* output, connx_Tensor* input) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);

	uint32_t output_size = connx_DataType_size(output->elemType);
	uint32_t input_size = connx_DataType_size(input->elemType);

	uint32_t total = output_total < input_total ? output_total : input_total;
	uint32_t size = output_size < input_size ? output_size : input_size;

	memcpy(output->base, input->base, total * size);

	return true;
}

#define CAST(output, output_type, input, input_type) 							\
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

// unit8
static bool uint8_to_uint16(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, uint8_t, input, uint16_t)
}

static bool uint8_to_uint32(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, uint8_t, input, uint32_t)
}

static bool uint8_to_uint64(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, uint8_t, input, uint64_t)
}

static bool uint8_to_int8(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, uint8_t, input, int8_t)
}

static bool uint8_to_int16(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, uint8_t, input, int16_t)
}

static bool uint8_to_int32(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, uint8_t, input, int32_t)
}

static bool uint8_to_int64(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, uint8_t, input, int64_t)
}

static bool uint8_to_float16(connx_Tensor* output, connx_Tensor* input) {
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

static bool uint8_to_float32(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, uint8_t, input, float)
}

static bool uint8_to_float64(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, uint8_t, input, double)
}

static bool uint8_to_bool(connx_Tensor* output, connx_Tensor* input) {
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

static bool uint8_to_string(connx_Tensor* output, connx_Tensor* input) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint8_t* input_base = (uint8_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%u", *input_base++);
		char* buf2 = malloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

// float32
static bool float32_to_uint8(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, float, input, uint8_t)
}

static bool float32_to_uint16(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, float, input, uint16_t)
}

static bool float32_to_uint32(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, float, input, uint32_t)
}

static bool float32_to_uint64(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, float, input, uint64_t)
}

static bool float32_to_int8(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, float, input, int8_t)
}

static bool float32_to_int16(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, float, input, int16_t)
}

static bool float32_to_int32(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, float, input, int32_t)
}

static bool float32_to_int64(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, float, input, int64_t)
}

static bool float32_to_float16(connx_Tensor* output, connx_Tensor* input) {
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

static bool float32_to_float64(connx_Tensor* output, connx_Tensor* input) {
	CAST(output, float, input, double)
}

static bool float32_to_bool(connx_Tensor* output, connx_Tensor* input) {
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

static bool float32_to_string(connx_Tensor* output, connx_Tensor* input) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	float* input_base = (float*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%f", *input_base++);
		char* buf2 = malloc(len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}


static bool Cast_exec(uintptr_t* stack) {
	connx_Tensor* output = (void*)stack[1];
	connx_Tensor* input = (void*)stack[2];
	__attribute__((unused)) int64_t* to = (void*)stack[3];

	switch(input->elemType) {
		case connx_DataType_UINT8:
			switch(input->elemType) {
				case connx_DataType_UINT8:
					return copy(output, input);
				case connx_DataType_UINT16:
					return uint8_to_uint16(output, input);
				case connx_DataType_UINT32:
					return uint8_to_uint32(output, input);
				case connx_DataType_UINT64:
					return uint8_to_uint64(output, input);
				case connx_DataType_INT8:
					return uint8_to_int8(output, input);
				case connx_DataType_INT16:
					return uint8_to_int16(output, input);
				case connx_DataType_INT32:
					return uint8_to_int32(output, input);
				case connx_DataType_INT64:
					return uint8_to_int64(output, input);
				case connx_DataType_FLOAT16:
					return uint8_to_float16(output, input);
				case connx_DataType_FLOAT32:
					return uint8_to_float32(output, input);
				case connx_DataType_FLOAT64:
					return uint8_to_float64(output, input);
				case connx_DataType_BOOL:
					return uint8_to_bool(output, input);
				case connx_DataType_STRING:
					return uint8_to_string(output, input);
				default:
					return false;
			}
			break;
		case connx_DataType_UINT16:
		case connx_DataType_UINT32:
		case connx_DataType_UINT64:
		case connx_DataType_INT8:
		case connx_DataType_INT16:
		case connx_DataType_INT32:
		case connx_DataType_INT64:
		case connx_DataType_FLOAT16:
		case connx_DataType_FLOAT32:
			switch(input->elemType) {
				case connx_DataType_UINT8:
					return float32_to_uint8(output, input);
				case connx_DataType_UINT16:
					return float32_to_uint16(output, input);
				case connx_DataType_UINT32:
					return float32_to_uint32(output, input);
				case connx_DataType_UINT64:
					return float32_to_uint64(output, input);
				case connx_DataType_INT8:
					return float32_to_int8(output, input);
				case connx_DataType_INT16:
					return float32_to_int16(output, input);
				case connx_DataType_INT32:
					return float32_to_int32(output, input);
				case connx_DataType_INT64:
					return float32_to_int64(output, input);
				case connx_DataType_FLOAT16:
					return float32_to_float16(output, input);
				case connx_DataType_FLOAT32:
					return copy(output, input);
				case connx_DataType_FLOAT64:
					return float32_to_float64(output, input);
				case connx_DataType_BOOL:
					return float32_to_bool(output, input);
				case connx_DataType_STRING:
					return float32_to_string(output, input);
				default:
					return false;
			}
			break;
		case connx_DataType_FLOAT64:
		case connx_DataType_BOOL:
		case connx_DataType_STRING:
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
