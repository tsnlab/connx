#include <stdlib.h>
#include <connx/connx.h>

static bool Loop_resolve(__attribute__((unused)) uintptr_t* stack) {
	/*
	uintptr_t count = stack[0];

	uintptr_t output_count = stack[2];
	connx_Value** v_final_and_scan_outputs = (void*)(stack + 3);

	uintptr_t input_count = stack[3 + output_count];
	char* M = (char*)stack[4 + output_count];
	char* cond = (char*)stack[5 + output_count];
	connx_Value** v_initials = (void*)(stack + 5 + output_count);

	void* graph = stack[count];
	*/

	return true;
}

static bool Loop_exec(__attribute__((unused)) uintptr_t* stack) {
	return true;
}

bool connx_opset_Loop_init() {
	connx_Operator_add("Loop", 1 | CONNX_VARARGS, 2 | CONNX_VARARGS, 1, Loop_resolve, Loop_exec,
		connx_DataType_ANY,		// v_final_and_scan_outputs
		connx_DataType_STRING,	// M
		connx_DataType_STRING,	// cond
								// v_initial
		"body", connx_DataType_GRAPH, NULL);

	return true;
}
