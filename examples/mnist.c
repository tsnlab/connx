#include <stdio.h>
#include <inttypes.h>
#include <sys/time.h>
#include <connx/connx.h>

static uint64_t get_us() {
	struct timeval  tv;
	gettimeofday(&tv, NULL);

	return (tv.tv_sec) * 1000000 + (tv.tv_usec);
}

int main(int argc, char** argv) {
	connx_init(-1);

	connx_Model* model = connx_Model_create_from_file("mnist.onnx");

	connx_Value* inputs[1];
	inputs[0] = (connx_Value*)connx_Tensor_create_from_file("mnist_input.pb");
	inputs[0]->name = "Input3";

	connx_Runtime* runtime = connx_Runtime_create(model);

	connx_Runtime_init(runtime);

	uint64_t time_start = get_us();
	for(uint32_t i = 0; i < 1000; i++) {
		connx_Runtime_run(runtime, 1, (connx_Value**)inputs);
	}
	uint64_t time_end = get_us();

	printf("Time: %" PRIu64 " us\n", time_end - time_start);

	connx_Runtime_delete(runtime);
	connx_SubThread_finalize();

	return 0;
}
