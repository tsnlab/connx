#include <stdio.h>
#include <connx/connx.h>
#include <connx/dump.h>

extern connx_HAL* hal_create(char* path);
extern void hal_delete(connx_HAL* hal);

int main(__attribute__((unused)) int argc, __attribute__((unused)) char** argv) {
	connx_HAL* hal = hal_create("out");

	// Make CONNX backend
	connx_Backend* backend = connx_Backend_create(hal);
	if(backend == NULL) {
		return -1;
	}

	// Load input tensors
	connx_Tensor* inputs[16] = { NULL, };
	for(int i = 0; i < 15 && i + 1 < argc; i++) {
		inputs[i] = connx_Backend_load_tensor(backend, argv[1 + i]);
	}

	// run
	connx_Tensor** outputs = connx_Backend_run(backend, inputs);
	if(outputs == NULL) {
		return -1;
	}

	printf("output in main: %p\n", outputs);
	for(uint32_t i = 0; outputs[i] != NULL; i++) {
		connx_Tensor_dump(hal, outputs[i]);
	}

	// clean
	connx_Backend_clean(backend);
	connx_Backend_delete(backend);

	hal_delete(hal);

	return 0;
}
