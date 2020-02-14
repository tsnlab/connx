#include <string.h>
#include <connx/connx.h>

static bool Shape_resolve(uintptr_t* stack) {
	connx_Tensor* shape = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];

	if(shape->dimension != 1) {
		connx_exception("shape's dimension must be 1 but: %u", shape->dimension);
		return false;
	}

	if(shape->lengths[0] != data->dimension) {
		connx_exception("shape's length is wrong: %u, expected: %u", shape->lengths[0], data->dimension);
		return false;
	}

	return true;
}

static bool Shape_exec(uintptr_t* stack) {
	connx_Tensor* shape = (void*)stack[1];
	connx_Tensor* data = (void*)stack[2];

	int64_t* shape_base = (int64_t*)shape->base;

	for(uint32_t i = 0; i < data->dimension; i++) {
		shape_base[i] = data->lengths[i];
	}

	return true;
}

bool connx_opset_Shape_init() {
	connx_Operator_add("Shape", 1, 1, 0, Shape_resolve, Shape_exec,
		connx_DataType_TENSOR_INT64,	// shape
		connx_DataType_TENSOR_NUMBER | connx_DataType_STRING | connx_DataType_BOOL);	// data

	return true;
}
