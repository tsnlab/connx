#include <string.h>
#include <connx/connx.h>

static bool Squeeze_resolve(uintptr_t* stack) {
	connx_Tensor* squeezed = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];
	void* attr_axes  = (void*)stack[3];

	uint32_t axes_length = connx_Attribute_length(attr_axes);
	int64_t* axes = connx_Attribute_base(attr_axes);

	// Normalize axes
	for(uint32_t i = 0; i < axes_length; i++) {
		if(axes[i] < 0)
			axes[i] += data->dimension;

		if(axes[i] < 0 || axes[i] >= data->dimension) {
			connx_exception("axes[%u]'s index is out of bounds: %u", i, axes[i]);
			return false;
		}

		if(data->lengths[axes[i]] != 1) {
			connx_exception("data[%u]'s lengths is not 1 but %u", axes[i], data->lengths[axes[i]]);
			return false;
		}
	}

	// Create squeezed if NULL
	if(squeezed == NULL) {
		uint32_t lengths[data->dimension - axes_length];
		for(uint32_t i = 0, idx = 0; i < data->dimension; i++) {
			bool has_axes = false;
			for(uint32_t j = 0; j < axes_length; j++) {
				if(i == axes[j]) {
					has_axes = true;
					break;
				}
			}

			if(!has_axes)
				lengths[idx++] = data->lengths[i];
		}

		squeezed = connx_Tensor_create2(data->elemType, data->dimension - axes_length, lengths);
		connx_Stack_update(1, squeezed);
	}

	if(axes_length > data->dimension) {
		connx_exception("axes's length cannot exceed data's dimension: %u > %u", axes_length, data->dimension);
		return false;
	}

	if(squeezed->dimension != data->dimension - axes_length) {
		connx_exception("squeezed's dimension is wrong: %u, expected: %u", squeezed->dimension, data->dimension - axes_length);
		return false;
	}

	if(squeezed->elemType != data->elemType) {
		connx_exception("data and squeezed's element type is different: %u != %u", data->elemType, squeezed->elemType);
		return false;
	}

	return true;
}

static bool Squeeze_exec(uintptr_t* stack) {
	connx_Tensor* squeezed = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];

	memcpy(squeezed->base, data->base, connx_Tensor_total(data) * connx_DataType_size(data->elemType));

	return true;
}

bool connx_opset_Squeeze_init() {
	connx_Operator_add("Squeeze", 1, 1, 1, Squeeze_resolve, Squeeze_exec,
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// squeezed
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL,	// data
		"axes", connx_DataType_INT64_ARRAY, 0, NULL);	// axes

	return true;
}
