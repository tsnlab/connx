#include <stdio.h>
#include <strings.h>
#include <sys/time.h>
#include <connx/connx.h>
#include <connx/dump.h>

static uint64_t get_us() {
	struct timeval  tv;
	gettimeofday(&tv, NULL);

	return (tv.tv_sec) * 1000000 + (tv.tv_usec);
}

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
	uint32_t input_count = argc - 1;
	connx_Tensor* inputs[input_count];
	for(uint32_t i = 0; i < input_count; i++) {
		inputs[i] = connx_Backend_load_tensor(backend, argv[1 + i]);
		if(inputs[i] == NULL) {
			return -1;
		}
	}

	// run
	uint32_t output_count = 16;
	connx_Tensor* outputs[output_count];
	bzero(outputs, sizeof(connx_Tensor*) * output_count);

	uint64_t base = get_us();

	for(uint32_t i = 0; i < 1000; i++) {
		output_count = 16;

		for(uint32_t j = 0; j < output_count; j++) {
			if(outputs[j] != NULL) {
				connx_Tensor_delete(hal, outputs[j]);
			}
		}

		if(!connx_Backend_run(backend, &output_count, outputs, input_count, inputs)) {
			connx_Backend_delete(backend);
			hal_delete(hal);
			return -1;
		}
	}

	printf("Time: %lu us\n", get_us() - base);

	for(uint32_t i = 0; i < output_count; i++) {
		connx_Tensor_dump(hal, outputs[i]);
		connx_Tensor_delete(hal, outputs[i]);
	}

	connx_Tensor* exptected = connx_Backend_load_tensor(backend, "output_0.tensor");
	connx_Tensor_dump(hal, exptected);
	connx_Tensor_delete(backend->hal, exptected);

	// clean
	for(uint32_t i = 0; i < input_count; i++)
		connx_Tensor_delete(backend->hal, inputs[i]);
	connx_Backend_delete(backend);

	hal_delete(hal);

	return 0;
}
