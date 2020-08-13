#include <inttypes.h>
#include <string.h>
#include <connx/operator.h>
#include <connx/backend.h>

bool opset_Shape(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* shape = CONNX_GET_OUTPUT(0);
	connx_Tensor* data = CONNX_GET_INPUT(0);

	// Create shape if NULL
	if(shape == NULL) {
		uint32_t lengths[1] = { data->dimension };

		shape = connx_Tensor_create(backend->pal, connx_INT64, 1, lengths);
		CONNX_SET_OUTPUT(0, shape);
	}

	int64_t* shape_base = (int64_t*)shape->base;

	for(uint32_t i = 0; i < data->dimension; i++) {
		shape_base[i] = data->lengths[i];
	}

	return true;
}
