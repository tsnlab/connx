#include <string.h>
#include <connx/connx.h>

static bool Conv_resolve(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	connx_Tensor* W = (void*)stack[3];
	void* attr_auto_pad = (void*)stack[4];
	void* attr_dilations = (void*)stack[5];
	void* attr_group = (void*)stack[6];
	void* attr_kernel_shape = (void*)stack[7];
	void* attr_pads = (void*)stack[8];
	void* attr_strides = (void*)stack[9];

	char* auto_pad = attr_auto_pad;
	uint32_t dilations_length = connx_Attribute_length(attr_dilations);
	int64_t* group = attr_group;
	int64_t* kernel_shape = connx_Attribute_base(attr_kernel_shape);
	uint32_t kernel_shape_length = connx_Attribute_length(attr_kernel_shape);
	int64_t* pads = connx_Attribute_base(attr_pads);
	uint32_t pads_length = connx_Attribute_length(attr_pads);
	int64_t* strides = connx_Attribute_base(attr_strides);
	uint32_t strides_length = connx_Attribute_length(attr_strides);

	// Create Y if NULL
	if(Y == NULL) {
		uint32_t lengths[X->dimension];
		memcpy(lengths, X->lengths, sizeof(uint32_t) * X->dimension);
		lengths[1] = W->lengths[0] * *group;

		Y = connx_Tensor_create2(X->elemType, X->dimension, lengths);
		stack[1] = (uintptr_t)Y;
	}

	if(auto_pad[0] == 'S') {	// SAME_UPPER, SAME_LOWER
		int64_t array[kernel_shape_length * 2];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			if(kernel_shape[i] > 2) {
				array[i] = array[i + kernel_shape_length] = (kernel_shape[i] - 1) / 2;
			} else {
				array[i] = array[i + kernel_shape_length] = X->lengths[i];;
			}
		}

		connx_Attribute_delete(attr_pads);
		stack[8] = connx_Attribute_create_ints(kernel_shape_length * 2, array);
		attr_pads = (void*)stack[8];
		pads = connx_Attribute_base(attr_pads);
		pads_length = connx_Attribute_length(attr_pads);

		/*
		printf("X: ");
		for(int i = 0; i < X->dimension; i++)
			printf("%lu ", X->lengths[i]);
		printf("\n");

		printf("kernel_shape: ");
		for(int i = 0; i < kernel_shape_length; i++)
			printf("%lu ", kernel_shape[i]);
		printf("\n");

		printf("pads: ");
		for(int i = 0; i < pads_length; i++)
			printf("%lu ", pads[i]);
		printf("\n");
		*/
	}

	if(dilations_length == 0) {
		int64_t array[kernel_shape_length];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			array[i] = 0;
		}

		connx_Attribute_delete(attr_dilations);
		stack[5] = connx_Attribute_create_ints(kernel_shape_length, array);
		attr_dilations = (void*)stack[5];
		dilations_length = connx_Attribute_length(attr_dilations);
	}

	if(strides_length == 0) {
		int64_t array[kernel_shape_length];
		for(uint32_t i = 0; i < kernel_shape_length; i++) {
			array[i] = 1;
		}

		connx_Attribute_delete(attr_strides);
		stack[9] = connx_Attribute_create_ints(kernel_shape_length, array);
		attr_strides = (void*)stack[9];
		strides = connx_Attribute_base(attr_strides);
		strides_length = connx_Attribute_length(attr_strides);
	}

	if(kernel_shape_length * 2 != pads_length) {
		connx_exception("pads shape is not maching: kernel_shape's dimension: %u but pads dimension: %u", kernel_shape_length, pads_length);
		return false;
	}

	if(X->dimension != kernel_shape_length + 2) {
		connx_exception("X's dimension: %u is and kernel_shape's dimension: %u is not matching", X->dimension, kernel_shape_length);
		return false;
	}

	if(W->dimension != kernel_shape_length + 2) {
		connx_exception("W's dimension: %u is and kernel_shape's dimension: %u is not matching", W->dimension, kernel_shape_length);
		return false;
	}

	if(kernel_shape_length != 2 && kernel_shape_length != 3) {
		connx_exception("kernel_shape count must be 2 or 3 but %u", kernel_shape_length);
		return false;
	}

	if(kernel_shape_length != dilations_length) {
		connx_exception("dilation shape dimension: %u is different to kernel_shape's dimension: %u", dilations_length, kernel_shape_length);
		return false;
	}

	if(kernel_shape_length != strides_length) {
		connx_exception("stride shape dimension: %u is different to kernel_shape's dimension: %u", strides_length, kernel_shape_length);
		return false;
	}

	if(Y->lengths[0] != X->lengths[0]) {
		connx_exception("Y's batch size is not matching: Y: %u but X: %u", Y->lengths[0], X->lengths[0]);
		return false;
	}

	if(Y->lengths[1] != W->lengths[0] * *group) {
		connx_exception("Y's feature size is not matching: Y: %u but W: %u", Y->lengths[1], W->lengths[1]);
		return false;
	}

	for(uint32_t i = 0; i < kernel_shape_length; i++) {
		if(Y->lengths[i + 2] != (X->lengths[i + 2] - kernel_shape[i] + pads[i] + pads[i + kernel_shape_length]) / strides[i] + 1) {
			connx_exception("Y's %uth shape is not matching: Y: %u, expected: %u", i + 2,
					Y->lengths[i + 2],
					(X->lengths[i + 2] - kernel_shape[i] + pads[i] + pads[i + kernel_shape_length]) / strides[i] + 1);
			return false;
		}
	}

	return true;
}

static void _conv2d_float(__attribute__((unused)) uint32_t* Y_lengths, float* Y, uint32_t* X_lengths, float* X, uint32_t* W_lengths, float* W, int64_t* kernels, int64_t* pads, int64_t* strides) {
	for(int64_t y = -pads[0]; y <= (int64_t)X_lengths[0] + pads[0 + 2] - kernels[0]; y += strides[0]) {
		for(int64_t x = -pads[1]; x <= (int64_t)X_lengths[1] + pads[1 + 2] - kernels[1]; x += strides[1]) {
			float tmp = 0;

			for(uint32_t ky = 0; ky < kernels[0]; ky++) {
				for(uint32_t kx = 0; kx < kernels[1]; kx++) {
					int64_t y2 = y + ky;
					int64_t x2 = x + kx;

					if(y2 >= 0 && x2 >= 0 && y2 < X_lengths[0] && x2 < X_lengths[1]) {
						tmp += X[y2 * X_lengths[1] + x2] * W[ky * W_lengths[1] + kx];
					}
				}
			}

			*Y++ += tmp;
		}
	}
}

static bool Conv_exec(uintptr_t* stack) {
	connx_Tensor* Y = (void*)stack[1];
	connx_Tensor* X = (void*)stack[2];
	connx_Tensor* W = (void*)stack[3];
	__attribute__((unused)) char* attr_auto_pad = (void*)stack[4];
	void* attr_dilations = (void*)stack[5];
	void* attr_group = (void*)stack[6];
	void* attr_kernel_shape = (void*)stack[7];
	void* attr_pads = (void*)stack[8];
	void* attr_strides = (void*)stack[9];

	__attribute__((unused)) uint32_t dilations_length = connx_Attribute_length(attr_dilations);
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
		y_unit *= Y->lengths[Y->dimension - 2 + i];
		x_unit *= X->lengths[X->dimension - 2 + i];
		w_unit *= W->lengths[W->dimension - 2 + i];
	}

	switch(X->elemType) {
		case connx_DataType_FLOAT16:
		case connx_DataType_FLOAT32:
			{
				float* y_array = (float*)Y->base;

				for(uint32_t batch = 0; batch < X->lengths[0]; batch++) {
					for(uint32_t feature = 0; feature < W->lengths[0]; feature++) {
						for(uint32_t channel = 0; channel < W->lengths[1] * *group; channel++) {
							float* x_array = (float*)X->base + batch * X->lengths[1] * x_unit + channel * x_unit;
							float* w_array = (float*)W->base + feature * W->lengths[1] * w_unit + channel / *group * w_unit;
							_conv2d_float(Y->lengths + Y->dimension - 2, y_array, 
									X->lengths + X->dimension - 2, x_array, 
									W->lengths + W->dimension - 2, w_array, 
									kernel_shape, pads, strides);

						}

						y_array += y_unit;
					}
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
	connx_Operator_add("Conv", 1, 2, 6, Conv_resolve, Conv_exec,
		connx_DataType_TENSOR_FLOAT,
		connx_DataType_TENSOR_FLOAT,
		connx_DataType_TENSOR_FLOAT,
		"auto_pad", connx_DataType_STRING, "NOTSET",
		"dilations", connx_DataType_INT64_ARRAY, 0, NULL, 
		"group", connx_DataType_INT64, 1,
		"kernel_shape", connx_DataType_INT64_ARRAY, 0, NULL, 
		"pads", connx_DataType_INT64_ARRAY, 0, NULL, 
		"strides", connx_DataType_INT64_ARRAY, 0, NULL);

	return true;
}
