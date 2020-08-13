#include <string.h>
#include <connx/operator.h>
#include <connx/backend.h>

bool opset_Squeeze(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* squeezed = CONNX_GET_OUTPUT(0);
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

	// Create squeezed if NULL
	if(squeezed == NULL) {
		uint32_t lengths[data->dimension - axes_length];
		for(uint32_t i = 0, idx = 0; i < data->dimension; i++) {
			bool has_axes = false;
			for(uint32_t j = 0; j < axes_length; j++) {
				if(i == axes_base[j]) {
					has_axes = true;
					break;
				}
			}

			if(!has_axes)
				lengths[idx++] = data->lengths[i];
		}

		squeezed = connx_Tensor_create(backend->pal, data->type, data->dimension - axes_length, lengths);
		CONNX_SET_OUTPUT(0, squeezed);
	}

	memcpy(squeezed->base, data->base, connx_Tensor_total(data) * connx_DataType_size(data->type));

	return true;
}
