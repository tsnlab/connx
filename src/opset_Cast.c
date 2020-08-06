#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <connx/operator.h>
#include <connx/backend.h>

static bool copy(connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);

	uint32_t output_size = connx_DataType_size(output->type);
	uint32_t input_size = connx_DataType_size(input->type);

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

// Ref: https://stackoverflow.com/questions/3026441/float32-to-float16/3026505
// Ref: https://gist.github.com/martin-kallman/5049614
// Ref: https://tool.oschina.net/uploads/apidocs/ogre3d/api/html/OgreBitwise_8h_source.html
static uint16_t connx_float32_to_float16(float v) {
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

static float connx_float16_to_float32(uint16_t v) {
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

static bool uint8_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint8_t* input_base = (uint8_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%" PRIu32, *input_base++);
		char* buf2 = hal->alloc(hal, len);
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

static bool uint16_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint16_t* input_base = (uint16_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%" PRIu32, *input_base++);
		char* buf2 = hal->alloc(hal, len);
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

static bool uint32_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint32_t* input_base = (uint32_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%" PRIu32, *input_base++);
		char* buf2 = hal->alloc(hal, len);
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

static bool uint64_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint64_t* input_base = (uint64_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%" PRIu64, *input_base++);
		char* buf2 = hal->alloc(hal, len);
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

static bool int8_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	int8_t* input_base = (int8_t*)input->base;
	char buf[32];
	for(int32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%" PRIu32, *input_base++);
		char* buf2 = hal->alloc(hal, len);
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

static bool int16_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	int16_t* input_base = (int16_t*)input->base;
	char buf[32];
	for(int32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%" PRIu32, *input_base++);
		char* buf2 = hal->alloc(hal, len);
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

static bool int32_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	int32_t* input_base = (int32_t*)input->base;
	char buf[32];
	for(int32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%" PRIu32, *input_base++);
		char* buf2 = hal->alloc(hal, len);
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

static bool int64_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	int32_t output_total = connx_Tensor_total(output);
	int32_t input_total = connx_Tensor_total(input);
	int32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	int64_t* input_base = (int64_t*)input->base;
	char buf[32];
	for(int32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%" PRIu64, *input_base++);
		char* buf2 = hal->alloc(hal, len);
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

static bool float16_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	uint16_t* input_base = (uint16_t*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%f", connx_float16_to_float32(*input_base++));
		char* buf2 = hal->alloc(hal, len);
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

static bool float32_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	float* input_base = (float*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%.9f", *input_base++);
		char* buf2 = hal->alloc(hal, len);
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

static bool float64_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	double* input_base = (double*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%.17f", *input_base++);
		char* buf2 = hal->alloc(hal, len);
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

static bool bool_to_string(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	double* input_base = (double*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%s", *input_base++ ? "true" : "false");
		char* buf2 = hal->alloc(hal, len);
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

static bool string_to_bool(connx_HAL* hal, connx_Tensor* input, connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	double* input_base = (double*)input->base;
	char buf[32];
	for(uint32_t i = 0; i < total; i++) {
		int len = snprintf(buf, 32, "%s", *input_base++ ? "true" : "false");
		char* buf2 = hal->alloc(hal, len);
		memcpy(buf2, buf, len);
		*output_base++ = buf2;
	}

	return true;
}

static bool string_to_string(connx_HAL* hal, connx_Tensor* input, __attribute__((unused)) connx_Tensor* output) {
	uint32_t output_total = connx_Tensor_total(output);
	uint32_t input_total = connx_Tensor_total(input);
	uint32_t total = output_total < input_total ? output_total : input_total;

	char** output_base = (char**)output->base;
	char** input_base = (char**)input->base;
	for(uint32_t i = 0; i < total; i++) {
		char* input = input_base[i];
		int len = strlen(input) + 1;
		char* buf = hal->alloc(hal, len);
		memcpy(buf, input, len);
		*output_base++ = buf;
	}

	return true;
}


bool opset_Cast(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* output = CONNX_GET_OUTPUT(0);
	connx_Tensor* input = CONNX_GET_INPUT(0);
	connx_AttributeInt* to = CONNX_GET_ATTRIBUTE(0);

	// Create output if null
	if(output == NULL) {
		output = connx_Tensor_create(backend->hal, to->value, input->dimension, input->lengths);
		CONNX_SET_OUTPUT(0, output);
	}

	// cast
	switch(input->type) {
		case connx_UINT8:
			switch(output->type) {
				case connx_UINT8:
					return copy(input, output);
				case connx_UINT16:
					return uint8_to_uint16(input, output);
				case connx_UINT32:
					return uint8_to_uint32(input, output);
				case connx_UINT64:
					return uint8_to_uint64(input, output);
				case connx_INT8:
					return uint8_to_int8(input, output);
				case connx_INT16:
					return uint8_to_int16(input, output);
				case connx_INT32:
					return uint8_to_int32(input, output);
				case connx_INT64:
					return uint8_to_int64(input, output);
				case connx_FLOAT16:
					return uint8_to_float16(input, output);
				case connx_FLOAT32:
					return uint8_to_float32(input, output);
				case connx_FLOAT64:
					return uint8_to_float64(input, output);
				case connx_BOOL:
					return uint8_to_bool(input, output);
				case connx_STRING:
					return uint8_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_UINT16:
			switch(output->type) {
				case connx_UINT8:
					return uint16_to_uint8(input, output);
				case connx_UINT16:
					return copy(input, output);
				case connx_UINT32:
					return uint16_to_uint32(input, output);
				case connx_UINT64:
					return uint16_to_uint64(input, output);
				case connx_INT8:
					return uint16_to_int8(input, output);
				case connx_INT16:
					return uint16_to_int16(input, output);
				case connx_INT32:
					return uint16_to_int32(input, output);
				case connx_INT64:
					return uint16_to_int64(input, output);
				case connx_FLOAT16:
					return uint16_to_float16(input, output);
				case connx_FLOAT32:
					return uint16_to_float32(input, output);
				case connx_FLOAT64:
					return uint16_to_float64(input, output);
				case connx_BOOL:
					return uint16_to_bool(input, output);
				case connx_STRING:
					return uint16_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_UINT32:
			switch(output->type) {
				case connx_UINT8:
					return uint32_to_uint8(input, output);
				case connx_UINT16:
					return uint32_to_uint16(input, output);
				case connx_UINT32:
					return copy(input, output);
				case connx_UINT64:
					return uint32_to_uint64(input, output);
				case connx_INT8:
					return uint32_to_int8(input, output);
				case connx_INT16:
					return uint32_to_int16(input, output);
				case connx_INT32:
					return uint32_to_int32(input, output);
				case connx_INT64:
					return uint32_to_int64(input, output);
				case connx_FLOAT16:
					return uint32_to_float16(input, output);
				case connx_FLOAT32:
					return uint32_to_float32(input, output);
				case connx_FLOAT64:
					return uint32_to_float64(input, output);
				case connx_BOOL:
					return uint32_to_bool(input, output);
				case connx_STRING:
					return uint32_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_UINT64:
			switch(output->type) {
				case connx_UINT8:
					return uint64_to_uint8(input, output);
				case connx_UINT16:
					return uint64_to_uint16(input, output);
				case connx_UINT32:
					return uint64_to_uint32(input, output);
				case connx_UINT64:
					return copy(input, output);
				case connx_INT8:
					return uint64_to_int8(input, output);
				case connx_INT16:
					return uint64_to_int16(input, output);
				case connx_INT32:
					return uint64_to_int32(input, output);
				case connx_INT64:
					return uint64_to_int64(input, output);
				case connx_FLOAT16:
					return uint64_to_float16(input, output);
				case connx_FLOAT32:
					return uint64_to_float32(input, output);
				case connx_FLOAT64:
					return uint64_to_float64(input, output);
				case connx_BOOL:
					return uint64_to_bool(input, output);
				case connx_STRING:
					return uint64_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_INT8:
			switch(output->type) {
				case connx_UINT8:
					return copy(input, output);
				case connx_UINT16:
					return int8_to_uint16(input, output);
				case connx_UINT32:
					return int8_to_uint32(input, output);
				case connx_UINT64:
					return int8_to_uint64(input, output);
				case connx_INT8:
					return int8_to_int8(input, output);
				case connx_INT16:
					return int8_to_int16(input, output);
				case connx_INT32:
					return int8_to_int32(input, output);
				case connx_INT64:
					return int8_to_int64(input, output);
				case connx_FLOAT16:
					return int8_to_float16(input, output);
				case connx_FLOAT32:
					return int8_to_float32(input, output);
				case connx_FLOAT64:
					return int8_to_float64(input, output);
				case connx_BOOL:
					return int8_to_bool(input, output);
				case connx_STRING:
					return int8_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_INT16:
			switch(output->type) {
				case connx_UINT8:
					return int16_to_uint8(input, output);
				case connx_UINT16:
					return copy(input, output);
				case connx_UINT32:
					return int16_to_uint32(input, output);
				case connx_UINT64:
					return int16_to_uint64(input, output);
				case connx_INT8:
					return int16_to_int8(input, output);
				case connx_INT16:
					return int16_to_int16(input, output);
				case connx_INT32:
					return int16_to_int32(input, output);
				case connx_INT64:
					return int16_to_int64(input, output);
				case connx_FLOAT16:
					return int16_to_float16(input, output);
				case connx_FLOAT32:
					return int16_to_float32(input, output);
				case connx_FLOAT64:
					return int16_to_float64(input, output);
				case connx_BOOL:
					return int16_to_bool(input, output);
				case connx_STRING:
					return int16_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_INT32:
			switch(output->type) {
				case connx_UINT8:
					return int32_to_uint8(input, output);
				case connx_UINT16:
					return int32_to_uint16(input, output);
				case connx_UINT32:
					return copy(input, output);
				case connx_UINT64:
					return int32_to_uint64(input, output);
				case connx_INT8:
					return int32_to_int8(input, output);
				case connx_INT16:
					return int32_to_int16(input, output);
				case connx_INT32:
					return int32_to_int32(input, output);
				case connx_INT64:
					return int32_to_int64(input, output);
				case connx_FLOAT16:
					return int32_to_float16(input, output);
				case connx_FLOAT32:
					return int32_to_float32(input, output);
				case connx_FLOAT64:
					return int32_to_float64(input, output);
				case connx_BOOL:
					return int32_to_bool(input, output);
				case connx_STRING:
					return int32_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_INT64:
			switch(output->type) {
				case connx_UINT8:
					return int64_to_uint8(input, output);
				case connx_UINT16:
					return int64_to_uint16(input, output);
				case connx_UINT32:
					return int64_to_uint32(input, output);
				case connx_UINT64:
					return copy(input, output);
				case connx_INT8:
					return int64_to_int8(input, output);
				case connx_INT16:
					return int64_to_int16(input, output);
				case connx_INT32:
					return int64_to_int32(input, output);
				case connx_INT64:
					return int64_to_int64(input, output);
				case connx_FLOAT16:
					return int64_to_float16(input, output);
				case connx_FLOAT32:
					return int64_to_float32(input, output);
				case connx_FLOAT64:
					return int64_to_float64(input, output);
				case connx_BOOL:
					return int64_to_bool(input, output);
				case connx_STRING:
					return int64_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_FLOAT16:
			switch(output->type) {
				case connx_UINT8:
					return float16_to_uint8(input, output);
				case connx_UINT16:
					return float16_to_uint16(input, output);
				case connx_UINT32:
					return float16_to_uint32(input, output);
				case connx_UINT64:
					return float16_to_uint64(input, output);
				case connx_INT8:
					return float16_to_int8(input, output);
				case connx_INT16:
					return float16_to_int16(input, output);
				case connx_INT32:
					return float16_to_int32(input, output);
				case connx_INT64:
					return float16_to_int64(input, output);
				case connx_FLOAT16:
					return copy(input, output);
				case connx_FLOAT32:
					return float16_to_float32(input, output);
				case connx_FLOAT64:
					return float16_to_float64(input, output);
				case connx_BOOL:
					return float16_to_bool(input, output);
				case connx_STRING:
					return float16_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_FLOAT32:
			switch(output->type) {
				case connx_UINT8:
					return float32_to_uint8(input, output);
				case connx_UINT16:
					return float32_to_uint16(input, output);
				case connx_UINT32:
					return float32_to_uint32(input, output);
				case connx_UINT64:
					return float32_to_uint64(input, output);
				case connx_INT8:
					return float32_to_int8(input, output);
				case connx_INT16:
					return float32_to_int16(input, output);
				case connx_INT32:
					return float32_to_int32(input, output);
				case connx_INT64:
					return float32_to_int64(input, output);
				case connx_FLOAT16:
					return float32_to_float16(input, output);
				case connx_FLOAT32:
					return copy(input, output);
				case connx_FLOAT64:
					return float32_to_float64(input, output);
				case connx_BOOL:
					return float32_to_bool(input, output);
				case connx_STRING:
					return float32_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_FLOAT64:
			switch(output->type) {
				case connx_UINT8:
					return float64_to_uint8(input, output);
				case connx_UINT16:
					return float64_to_uint16(input, output);
				case connx_UINT32:
					return float64_to_uint32(input, output);
				case connx_UINT64:
					return float64_to_uint64(input, output);
				case connx_INT8:
					return float64_to_int8(input, output);
				case connx_INT16:
					return float64_to_int16(input, output);
				case connx_INT32:
					return float64_to_int32(input, output);
				case connx_INT64:
					return float64_to_int64(input, output);
				case connx_FLOAT16:
					return float64_to_float16(input, output);
				case connx_FLOAT32:
					return float64_to_float32(input, output);
				case connx_FLOAT64:
					return copy(input, output);
				case connx_BOOL:
					return float64_to_bool(input, output);
				case connx_STRING:
					return float64_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_BOOL:
			switch(output->type) {
				case connx_UINT8:
					return bool_to_uint8(input, output);
				case connx_UINT16:
					return bool_to_uint16(input, output);
				case connx_UINT32:
					return bool_to_uint32(input, output);
				case connx_UINT64:
					return bool_to_uint64(input, output);
				case connx_INT8:
					return bool_to_int8(input, output);
				case connx_INT16:
					return bool_to_int16(input, output);
				case connx_INT32:
					return bool_to_int32(input, output);
				case connx_INT64:
					return bool_to_int64(input, output);
				case connx_FLOAT16:
					return bool_to_float16(input, output);
				case connx_FLOAT32:
					return bool_to_float32(input, output);
				case connx_FLOAT64:
					return bool_to_float64(input, output);
				case connx_BOOL:
					return copy(input, output);
				case connx_STRING:
					return bool_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		case connx_STRING:
			switch(output->type) {
				case connx_UINT8:
					return string_to_uint8(input, output);
				case connx_UINT16:
					return string_to_uint16(input, output);
				case connx_UINT32:
					return string_to_uint32(input, output);
				case connx_UINT64:
					return string_to_uint64(input, output);
				case connx_INT8:
					return string_to_int8(input, output);
				case connx_INT16:
					return string_to_int16(input, output);
				case connx_INT32:
					return string_to_int32(input, output);
				case connx_INT64:
					return string_to_int64(input, output);
				case connx_FLOAT16:
					return string_to_float16(input, output);
				case connx_FLOAT32:
					return string_to_float32(input, output);
				case connx_FLOAT64:
					return string_to_float64(input, output);
				case connx_BOOL:
					return string_to_bool(backend->hal, input, output);
				case connx_STRING:
					return string_to_string(backend->hal, input, output);
				default:
					return false;
			}
			break;
		default:
			return false;
	}

	return true;
}
