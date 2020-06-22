#include <string.h>
#include <connx/connx.h>

static bool hasAxis(int64_t idx, int64_t* axes, uint32_t length, uint32_t dimension) {
	for(uint32_t i = 0; i < length; i++) {
		if((axes[i] < 0 && (axes[i] + dimension) == idx) || axes[i] == idx) {
			return true;
		}
	}

	return false;
}

static bool Unsqueeze_resolve(uintptr_t* stack) {
	connx_Tensor* expanded = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];
	void* attr_axes  = (void*)stack[3];

	uint32_t axes_length = connx_Attribute_length(attr_axes);
	int64_t* axes = connx_Attribute_base(attr_axes);

	// Create expanded if null
	if(expanded == NULL) {
		uint32_t len = data->dimension + axes_length;
		uint32_t lengths[len];

		for(uint32_t i = 0, data_idx = 0, axes_idx = 0; i < len; i++) {
			if(hasAxis(axes_idx, axes, axes_length, data->dimension)) {
				lengths[i] = 1;
				axes_idx++;
			} else {
				lengths[i] = data->lengths[data_idx++];
			}
		}

		expanded = connx_Tensor_create2(data->elemType, len, lengths);
		connx_Operator_stack_update(expanded, 1, 1);
	}

	if(expanded->dimension != data->dimension + axes_length) {
		connx_exception("expanded's dimension is wrong: %u, expected: %u", expanded->dimension, data->dimension + axes_length);
		return false;
	}

	if(expanded->elemType != data->elemType) {
		connx_exception("data and expanded's element type is different: %u != %u", data->elemType, expanded->elemType);
		return false;
	}

	for(uint32_t i = 0; i < axes_length; i++) {
		if(axes[i] < 0)
			axes[i] += data->dimension;

		if(axes[i] < 0 || axes[i] >= expanded->dimension) {
			connx_exception("axes[%u]'s index is out of bounds: %u", i, axes[i]);
			return false;
		}

		if(expanded->lengths[axes[i]] != 1) {
			connx_exception("expanded[%u]'s lengths is not 1: %u", axes[i], expanded->lengths[axes[i]]);
			return false;
		}
	}

	return true;
}

static bool Unsqueeze_exec(uintptr_t* stack) {
	connx_Tensor* expanded = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];

	memcpy(expanded->base, data->base, connx_Tensor_total(data) * connx_DataType_size(data->elemType));

	return true;
}

bool connx_opset_Unsqueeze_init() {
	connx_Operator_add("Unsqueeze", 1, 1, 1, Unsqueeze_resolve, Unsqueeze_exec,
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// expanded
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// data
		"axes", connx_DataType_INT64_ARRAY, 0, NULL);	// axes

	return true;
}
