#include <stdio.h>
#include <sys/time.h>
#include <connx/connx.h>

uint64_t get_time() {
	struct timeval  tv;
	gettimeofday(&tv, NULL);

	return (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000;
}

int main(__attribute__((unused)) int argc, __attribute__((unused)) char** argv) {
	printf("* init\n");
	connx_init();

	printf("* input\n");
	connx_Tensor* input = connx_Tensor_create_from_file("examples/mnist/test_data_set_0/input_0.pb");
	connx_Tensor_dump(input);

	printf("* output\n");
	connx_Tensor* output = connx_Tensor_create_from_file("examples/mnist/test_data_set_0/output_0.pb");
	connx_Tensor_dump(output);

	onnx_ModelProto* model = onnx_Model_create_from_file("examples/mnist/model.onnx");
	onnx_Model_dump(model);

	printf("* operators\n");
	connx_Operator_dump();

	connx_Runtime* runtime = connx_Runtime_create(model);
	connx_Value* result;
	uint64_t time_start = get_time();
	for(uint32_t i = 0; i < 1000; i++) {
		result = connx_Runtime_run(runtime, (connx_Value*)input);
	}
	uint64_t time_end = get_time();
	printf("%lu ms\n", time_end - time_start);

	printf("* result\n");
	if(result != NULL && result->type == connx_DataType_TENSOR) {
		connx_Tensor* tensor = (connx_Tensor*)result;
		connx_Tensor_dump(tensor);
	}

	printf("* output\n");
	connx_Tensor_dump(output);

	return 0;
}
