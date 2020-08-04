#include <string.h>
#include <strings.h>
#include <connx/operator.h>
#include <connx/backend.h>

static void _conv2d_float(__attribute__((unused)) uint32_t* Y_lengths, float* Y, uint32_t* X_lengths, float* X, uint32_t* W_lengths, float* W, int32_t* kernels, int32_t* pads, int32_t* strides, float bias) {
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

	uint32_t ky_end = kernels[0];
	uint32_t kx_end = kernels[1];
	uint32_t height = X_lengths[0];
	uint32_t width = X_lengths[1];
	uint32_t y_stride = strides[0];
	uint32_t x_stride = strides[1];
	int32_t y_start = -pads[0];
	int32_t y_end = height + pads[0 + 2] - ky_end;
	int32_t x_start = -pads[1];
	int32_t x_end = width + pads[1 + 2] - kx_end;
	uint32_t w_lengths_1 = W_lengths[1];

	for(int32_t y = y_start; y <= y_end; y += y_stride) {
		for(int32_t x = x_start; x <= x_end; x += x_stride) {
			float tmp = 0;

			uint32_t y2_start = MAX(y, 0);
			uint32_t y2_end = MIN(y + ky_end, height);

			uint32_t x2_start = MAX(x, 0);
			uint32_t x2_end = MIN(x + kx_end, width);

			for(uint32_t y2 = y2_start; y2 < y2_end; y2++) {
				uint32_t x_base = y2 * width;
				uint32_t ky = y2 - y;
				uint32_t w_base = ky * w_lengths_1 - x;

				for(uint32_t x2 = x2_start; x2 < x2_end; x2++) {
					tmp += X[x_base + x2] * W[w_base + x2];
				}
			}

			*Y++ += tmp + bias;
		}
	}
}

struct Work {
	uint32_t batch;
	uint32_t group;
	uint32_t feature;
};

struct Context {
	float* y_array;
	float* b_array;

	uint8_t* y_base;
	uint32_t y_unit;
	uint32_t* y_lengths_2;

	uint8_t* x_base;
	uint32_t x_unit;
	uint32_t* x_lengths_2;

	uint8_t* w_base;
	uint32_t w_unit;
	uint32_t w_lengths_1;
	uint32_t* w_lengths_2;

	uint32_t channel_count;
	uint32_t feature_count;
	uint32_t group_count;

	uint32_t group_unit;
	uint32_t batch_unit;

	int32_t* kernel_shape;
	int32_t* pads;
	int32_t* strides;
};

static void* run_float32(uint32_t work_count, void* _works, void* _context) {
	struct Work* works = _works;
	struct Context* context = _context;

	float* y_array = context->y_array;
	float* b_array = context->b_array;

	uint8_t* y_base = context->y_base;
	uint32_t y_unit = context->y_unit;
	uint32_t* y_lengths_2 = context->y_lengths_2;

	uint8_t* x_base = context->x_base;
	uint32_t x_unit = context->x_unit;
	uint32_t* x_lengths_2 = context->x_lengths_2;

	uint8_t* w_base = context->w_base;
	uint32_t w_unit = context->w_unit;
	uint32_t w_lengths_1 = context->w_lengths_1;
	uint32_t* w_lengths_2 = context->w_lengths_2;

	uint32_t channel_count = context->channel_count;
	uint32_t feature_count = context->feature_count;
	uint32_t group_count = context->group_count;

	uint32_t group_unit = context->group_unit;
	uint32_t batch_unit = context->batch_unit;

	int32_t* kernel_shape = context->kernel_shape;
	int32_t* pads = context->pads;
	int32_t* strides = context->strides;

	for(uint32_t i = 0; i < work_count; i++) {
		uint32_t batch = works[i].batch;
		uint32_t group = works[i].group;
		uint32_t feature = works[i].feature;

		y_array = (float*)y_base + batch * batch_unit + group * group_unit + feature * y_unit;

		for(uint32_t channel = 0; channel < channel_count; channel++) {
			uint32_t f = group * feature_count + feature;
			uint32_t c = group * channel_count + channel;

			float* x_array = (float*)x_base + (batch * channel_count * group_count + c) * x_unit;
			float* w_array = (float*)w_base + (f * w_lengths_1 + (c / group_count)) * w_unit;

			_conv2d_float(y_lengths_2, y_array, 
					x_lengths_2, x_array, 
					w_lengths_2, w_array, 
					kernel_shape, pads, strides, b_array != NULL ? b_array[feature] : 0);

		}
	}

	return NULL;
}

bool opset_Conv(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	uint32_t input_count = CONNX_INPUT_COUNT(counts);

	// outputs
	connx_Tensor* Y = CONNX_GET_OUTPUT(0);

	// inputs
	connx_Tensor* X = CONNX_GET_INPUT(0);
	connx_Tensor* W = CONNX_GET_INPUT(1);
	connx_Tensor* B = input_count >= 3 ? CONNX_GET_INPUT(2) : NULL;

	// attributes
	connx_AttributeString* auto_pad = CONNX_GET_ATTRIBUTE(0);
	connx_AttributeInts* dilations = CONNX_GET_ATTRIBUTE(1);
	connx_AttributeInt* group = CONNX_GET_ATTRIBUTE(2);
	connx_AttributeInts* kernel_shape = CONNX_GET_ATTRIBUTE(3);
	connx_AttributeInts* pads = CONNX_GET_ATTRIBUTE(4);
	connx_AttributeInts* strides = CONNX_GET_ATTRIBUTE(5);

	int32_t pad_values[kernel_shape->length * 2];
	bzero(pad_values, sizeof(int32_t) * kernel_shape->length * 2);

	if(auto_pad->value[0] == 'S') {
		for(uint32_t i = 0; i < kernel_shape->length; i++) {
			uint32_t input_shape = X->lengths[X->dimension - kernel_shape->length + i];
			uint32_t output_shape = input_shape / strides->values[i] + (input_shape % strides->values[i] > 0 ? 1 : 0);
			uint32_t pad = (output_shape - 1) * strides->values[i] + ((kernel_shape->values[i] - 1) * dilations->values[i] + 1) - input_shape;
			pad_values[i] = pad_values[i + kernel_shape->length] = pad / 2;
			if(pad % 2 == 1) {
				if(auto_pad->value[5] == 'U') {	// SAME_UPPER
					pad_values[i + kernel_shape->length]++;
				} else {						// SAME_LOWER
					pad_values[i]++;
				}
			}
		}
	} else if(auto_pad->value[0] == 'V') {
		bzero(pad_values, sizeof(int32_t) * kernel_shape->length * 2);
	} else {
		memcpy(pad_values, pads->values, sizeof(int32_t) * kernel_shape->length * 2);
	}

	if(Y == NULL) {
		// batch, feature * channel / group / filter, x1, x2, x3...
		uint32_t lengths[X->dimension];
		memcpy(lengths, X->lengths, sizeof(uint32_t) * X->dimension);
		lengths[1] = X->lengths[1] * W->lengths[0] / group->value / W->lengths[1];

		for(uint32_t i = 0; i < kernel_shape->length; i++) {
			lengths[i + 2] = (X->lengths[i + 2] - kernel_shape->values[i] + pad_values[i] + pad_values[i + kernel_shape->length]) / strides->values[i] + 1;
		}

		Y = connx_Tensor_create(backend->hal, X->type, X->dimension, lengths);

		CONNX_SET_OUTPUT(0, Y);
	}

	// make output tensor
	uint32_t x_unit = 1;
	uint32_t y_unit = 1;
	uint32_t w_unit = 1;
	for(uint32_t i = 0; i < kernel_shape->length; i++) {
		y_unit *= Y->lengths[2 + i];
		x_unit *= X->lengths[2 + i];
		w_unit *= W->lengths[2 + i];
	}

	switch(X->type) {
		case connx_FLOAT16:
		case connx_FLOAT32:
			{
				uint32_t batch_count = X->lengths[0];
				uint32_t channel_count = X->lengths[1] / group->value;
				uint32_t feature_count = W->lengths[0] / group->value;

				uint32_t group_unit = feature_count * y_unit;
				uint32_t batch_unit = group->value * group_unit;

				// Create works
				uint32_t work_count = batch_count * group->value * feature_count;
				struct Work works[work_count];
				uint32_t work_idx = 0;
				for(uint32_t batch = 0; batch < batch_count; batch++) {
					for(int32_t g = 0; g < group->value; g++) {
						for(uint32_t feature = 0; feature < feature_count; feature++) {
							works[work_idx].batch = batch;
							works[work_idx].group = g;
							works[work_idx].feature = feature;
							work_idx++;
						}
					}
				}

				struct Context context = {
					.y_array = (float*)Y->base,
					.b_array = B != NULL ? (float*)B->base : NULL,
                                                     
					.y_base = Y->base,
					.y_unit = y_unit,
					.y_lengths_2 = Y->lengths + 2,
                                                     
					.x_base = X->base,
					.x_unit = x_unit,
					.x_lengths_2 = X->lengths + 2,
                                                     
					.w_base = W->base,
					.w_unit = w_unit,
					.w_lengths_1 = W->lengths[1],
					.w_lengths_2 = W->lengths + 2,
                                                     
					.channel_count = channel_count,
					.feature_count = feature_count,
					.group_count = group->value,
                                                     
					.group_unit = group_unit,
					.batch_unit = batch_unit,
                                                     
					.kernel_shape = kernel_shape->values,
					.pads = pad_values,
					.strides = strides->values,
				};

				// Allocate threads
				//connx_SubThread* threads[work_count];
				//uint32_t thread_count = connx_SubThread_alloc(threads, work_count - 1);
				work_idx = 0;
				struct Work* ws = works;

				/*
				if(thread_count > 0) {
					uint32_t work_batch = (work_count / (thread_count + 1)) + (work_count % (thread_count + 1) > 0 ? 1 : 0);
					
					for(uint32_t i = 0; i < thread_count; i++) {
						uint32_t remain = work_count - work_idx;
						if(remain > work_batch)
							remain = work_batch;

						connx_SubThread_run(threads[i], run_float32, remain, ws, &context);

						ws += remain;
						work_idx += remain;
					}
				}
				*/

				run_float32(work_count - work_idx, ws, &context);

				/*
				for(uint32_t i = 0; i < thread_count; i++) {
					connx_SubThread_wait(threads[i]);
				}
				*/
			}
			break;
		case connx_FLOAT64:
			break;
		default:
			;
	}

	return true;
}
