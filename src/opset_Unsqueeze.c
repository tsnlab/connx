#include <string.h>
#include <connx/operator.h>
#include <connx/backend.h>

static bool has_axis(uint32_t idx, uint32_t* axes, uint32_t length, uint32_t dimension) {
	for(uint32_t i = 0; i < length; i++) {
		if((axes[i] < 0 && (axes[i] + dimension) == idx) || axes[i] == idx) {
			return true;
		}
	}

	return false;
}

bool opset_Unsqueeze(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* expanded = CONNX_GET_OUTPUT(0);
	connx_Tensor* data = CONNX_GET_INPUT(0);
	connx_AttributeInts* axes  = CONNX_GET_ATTRIBUTE(0);

	// Normalize axes
	uint32_t axes_length = axes->length;
	uint32_t axes_base[axes_length];

	for(uint32_t i = 0; i < axes_length; i++) {
		if(axes->values[i] < 0)
			axes_base[i] = (uint32_t)(axes->values[i] + data->dimension);
		else
			axes_base[i] = (uint32_t)axes->values[i];
	}

	// Create expanded if null
	if(expanded == NULL) {
		uint32_t len = data->dimension + axes_length;
		uint32_t lengths[len];

		for(uint32_t i = 0, data_idx = 0, axes_idx = 0; i < len; i++) {
			if(has_axis(axes_idx, axes_base, axes_length, data->dimension)) {
				lengths[i] = 1;
				axes_idx++;
			} else {
				lengths[i] = data->lengths[data_idx++];
			}
		}

		expanded = connx_Tensor_create(backend->hal, data->type, len, lengths);
		CONNX_SET_OUTPUT(0, expanded);
	}

	memcpy(expanded->base, data->base, connx_Tensor_total(data) * connx_DataType_size(data->type));

	return true;
}
