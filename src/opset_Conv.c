#include <inttypes.h>
#include <string.h>
#include <connx/connx.h>

static bool Conv_resolve(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	connx_Tensor* W = (void*)stack[3];
	connx_Tensor* B = (void*)stack[4];
	void* attr_auto_pad = (void*)stack[5];
	void* attr_dilations = (void*)stack[6];
	void* attr_group = (void*)stack[7];
	void* attr_kernel_shape = (void*)stack[8];
	void* attr_pads = (void*)stack[9];
	void* attr_strides = (void*)stack[10];

	char* auto_pad = attr_auto_pad;
	uint32_t dilations_length = connx_Attribute_length(attr_dilations);
	int64_t* dilations = connx_Attribute_base(attr_dilations);
	__attribute__((unused)) int64_t* group = attr_group;
	int64_t* kernel_shape = connx_Attribute_base(attr_kernel_shape);
	uint32_t kernel_shape_length = connx_Attribute_length(attr_kernel_shape);
	int64_t* pads = connx_Attribute_base(attr_pads);
	uint32_t pads_length = connx_Attribute_length(attr_pads);
	int64_t* strides = connx_Attribute_base(attr_strides);
	uint32_t strides_length = connx_Attribute_length(attr_strides);

	if(auto_pad[0] == 'S') {	// SAME_UPPER, SAME_LOWER
		int64_t array[kernel_shape_length * 2];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			// Same logic with MaxPool
			int64_t input_shape = X->lengths[X->dimension - kernel_shape_length + i];
			int64_t output_shape = input_shape / strides[i] + (input_shape % strides[i] > 0 ? 1 : 0);
			int64_t pad = (output_shape - 1) * strides[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - input_shape;
			array[i] = array[i + kernel_shape_length] = pad / 2;
			if(pad % 2 == 1) {
				if(auto_pad[5] == 'U') {	// SAME_UPPER
					array[i + kernel_shape_length]++;
				} else {					// SAME_LOWER
					array[i]++;
				}
			}
		}

		connx_Attribute_delete(attr_pads);
		stack[9] = connx_Attribute_create_ints(kernel_shape_length * 2, array);
		attr_pads = (void*)stack[9];
		pads = connx_Attribute_base(attr_pads);
		pads_length = connx_Attribute_length(attr_pads);
	}

	if(dilations_length == 0) {
		int64_t array[kernel_shape_length];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			array[i] = 0;
		}

		connx_Attribute_delete(attr_dilations);
		stack[6] = connx_Attribute_create_ints(kernel_shape_length, array);
		attr_dilations = (void*)stack[6];
		dilations_length = connx_Attribute_length(attr_dilations);
	}

	if(strides_length == 0) {
		int64_t array[kernel_shape_length];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			array[i] = 1;
		}

		connx_Attribute_delete(attr_strides);
		stack[10] = connx_Attribute_create_ints(kernel_shape_length, array);
		attr_strides = (void*)stack[10];
		strides = connx_Attribute_base(attr_strides);
		strides_length = connx_Attribute_length(attr_strides);
	}

	if(kernel_shape_length * 2 != pads_length) {
		connx_exception("pads shape is not maching: kernel_shape's dimension: %" PRIu32 " but pads dimension: %" PRIu32, kernel_shape_length, pads_length);
		return false;
	}

	if(X->dimension != kernel_shape_length + 2) {
		connx_exception("X's dimension: %" PRIu32 " and kernel_shape's dimension: %" PRIu32 " is not matching", X->dimension, kernel_shape_length);
		return false;
	}

	if(W->dimension != kernel_shape_length + 2) {
		connx_exception("W's dimension: %" PRIu32 " is and kernel_shape's dimension: %" PRIu32 " is not matching", W->dimension, kernel_shape_length);
		return false;
	}

	if(W->lengths[1] * *group != X->lengths[1]) {
		connx_exception("W's feature count is not matching: expected: %" PRIu32 " * %" PRIu32 " = %" PRIu32 " but %" PRIu32, W->lengths[1], *group, W->lengths[1] * *group, X->lengths[1]);
		return false;
	}

	if(kernel_shape_length != 2 && kernel_shape_length != 3) {
		connx_exception("kernel_shape count must be 2 or 3 but %" PRIu32, kernel_shape_length);
		return false;
	}

	if(kernel_shape_length != dilations_length) {
		connx_exception("dilation shape dimension: %" PRIu32 " is different to kernel_shape's dimension: %" PRIu32, dilations_length, kernel_shape_length);
		return false;
	}

	if(kernel_shape_length != strides_length) {
		connx_exception("stride shape dimension: %" PRIu32 " is different to kernel_shape's dimension: %" PRIu32, strides_length, kernel_shape_length);
		return false;
	}

	// Create Y if NULL
	if(Y == NULL) {
		// batch, feature * channel / group / filter, x1, x2, x3...
		uint32_t lengths[X->dimension];
		memcpy(lengths, X->lengths, sizeof(uint32_t) * X->dimension);
		lengths[1] = X->lengths[1] * W->lengths[0] / *group / W->lengths[1];

		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			lengths[i + 2] = (X->lengths[i + 2] - kernel_shape[i] + pads[i] + pads[i + kernel_shape_length]) / strides[i] + 1;
		}

		Y = connx_Tensor_create2(X->elemType, X->dimension, lengths);
		connx_Stack_update(1, Y);
	}

	if(Y->lengths[0] != X->lengths[0]) {
		connx_exception("Y's batch size is not matching: Y: %" PRIu32 " but X: %" PRIu32, Y->lengths[0], X->lengths[0]);
		return false;
	}

	if(Y->lengths[1] != W->lengths[0] * X->lengths[1] / *group / W->lengths[1]) {
		connx_exception("Y's feature size is not matching: Y[1]: %" PRIu32 " != W[0]: %" PRIu32 " * X[1]: %" PRIu32 " / group: %" PRIu32, Y->lengths[1], W->lengths[0], X->lengths[1], *group);
		return false;
	}

	for(uint32_t i = 0; i < kernel_shape_length; i++) {
		if(Y->lengths[i + 2] != (X->lengths[i + 2] - kernel_shape[i] + pads[i] + pads[i + kernel_shape_length]) / strides[i] + 1) {
			connx_exception("Y's %" PRIu32 "th shape is not matching: Y: %" PRIu32 ", expected: %" PRIu32, i + 2,
					Y->lengths[i + 2],
					(X->lengths[i + 2] - kernel_shape[i] + pads[i] + pads[i + kernel_shape_length]) / strides[i] + 1);
			return false;
		}
	}

	if(B != NULL) {
		if(B->elemType != X->elemType) {
			connx_exception("B's elemType is differ from X's: %" PRIu32 " != %" PRIu32, B->elemType, X->elemType);
			return false;
		}

		if(B->dimension != 1) {
			connx_exception("B's dimension must be 1 but %" PRIu32, B->dimension);
			return false;
		}

		if(B->lengths[0] == X->lengths[0]) {
			connx_exception("B's length must be equal to batch size: %" PRIu32 " != %" PRIu32, B->lengths[0], X->lengths[0]);
			return false;
		}
	}

	return true;
}

static void _conv2d_float(__attribute__((unused)) uint32_t* Y_lengths, float* Y, uint32_t* X_lengths, float* X, uint32_t* W_lengths, float* W, int64_t* kernels, int64_t* pads, int64_t* strides, float bias) {
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

	int64_t* kernel_shape;
	int64_t* pads;
	int64_t* strides;
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

	int64_t* kernel_shape = context->kernel_shape;
	int64_t* pads = context->pads;
	int64_t* strides = context->strides;

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

static bool Conv_exec(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	connx_Tensor* W = (void*)stack[3];
	connx_Tensor* B = (void*)stack[4];
	__attribute__((unused)) char* attr_auto_pad = (void*)stack[5];
	__attribute__((unused)) void* attr_dilations = (void*)stack[6];
	void* attr_group = (void*)stack[7];
	void* attr_kernel_shape = (void*)stack[8];
	void* attr_pads = (void*)stack[9];
	void* attr_strides = (void*)stack[10];

	int64_t* group = attr_group;
	int64_t* kernel_shape = connx_Attribute_base(attr_kernel_shape);
	uint32_t kernel_shape_length = connx_Attribute_length(attr_kernel_shape);
	int64_t* pads = connx_Attribute_base(attr_pads);
	int64_t* strides = connx_Attribute_base(attr_strides);

	// make output tensor
	uint32_t x_unit = 1;
	uint32_t y_unit = 1;
	uint32_t w_unit = 1;
	for(uint32_t i = 0; i < kernel_shape_length; i++) {
		y_unit *= Y->lengths[2 + i];
		x_unit *= X->lengths[2 + i];
		w_unit *= W->lengths[2 + i];
	}

	switch(X->elemType) {
		case connx_DataType_FLOAT16:
		case connx_DataType_FLOAT32:
			{
				uint32_t batch_count = X->lengths[0];
				uint32_t channel_count = X->lengths[1] / *group;
				uint32_t feature_count = W->lengths[0] / *group;

				uint32_t group_unit = feature_count * y_unit;
				uint32_t batch_unit = *group * group_unit;

				// Create works
				uint32_t work_count = batch_count * *group * feature_count;
				struct Work works[work_count];
				uint32_t work_idx = 0;
				for(uint32_t batch = 0; batch < batch_count; batch++) {
					for(uint32_t g = 0; g < *group; g++) {
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
					.group_count = *group,
                                                     
					.group_unit = group_unit,
					.batch_unit = batch_unit,
                                                     
					.kernel_shape = kernel_shape,
					.pads = pads,
					.strides = strides,
				};


				// Allocate threads
				connx_SubThread* threads[work_count];
				uint32_t thread_count = connx_SubThread_alloc(threads, work_count - 1);
				work_idx = 0;
				struct Work* ws = works;

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

				run_float32(work_count - work_idx, ws, &context);

				for(uint32_t i = 0; i < thread_count; i++) {
					connx_SubThread_wait(threads[i]);
				}
			}
			break;
		case connx_DataType_FLOAT64:
			break;
		default:
			;
	}

	return true;
}

bool connx_opset_Conv_init() {
	connx_Operator_add("Conv", 1, 3, 6, Conv_resolve, Conv_exec,
		connx_DataType_TENSOR_FLOAT,	// Y
		connx_DataType_TENSOR_FLOAT,	// X
		connx_DataType_TENSOR_FLOAT,	// W
		connx_DataType_TENSOR_FLOAT,	// B (optional)
		"auto_pad", connx_DataType_STRING, "NOTSET",
		"dilations", connx_DataType_INT64_ARRAY, 0, NULL, 
		"group", connx_DataType_INT64, 1,
		"kernel_shape", connx_DataType_INT64_ARRAY, 0, NULL, 
		"pads", connx_DataType_INT64_ARRAY, 0, NULL, 
		"strides", connx_DataType_INT64_ARRAY, 0, NULL);

	return true;
}
