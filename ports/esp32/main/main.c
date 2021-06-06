/* Hello World Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include <string.h>
#include <sys/unistd.h>
#include <connx/tensor.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_spi_flash.h"

#include <connx/connx.h>

void app_main(void) {
	connx_init();

    // Parse connx model
    connx_Model model;
    int ret = connx_Model_init(&model);
    if(ret != 0) {
		connx_error("Cannot load model");
        return;
    }

    void* buf = connx_load("test_data_set_0/input_0.data");
    connx_Tensor* tensor = connx_Tensor_alloc_buffer(buf);
    connx_unload(buf);
    //int32_t shape[] = {1, 1, 28, 28};
    //connx_Tensor* tensor = connx_Tensor_alloc(CONNX_FLOAT32, 4, shape);
    //float* array = (float*)tensor->buffer;
    //connx_Tensor_dump(tensor);

	uint32_t input_count = 1;
	connx_Tensor* inputs[] = {tensor};

	uint32_t output_count = 16;
	connx_Tensor* outputs[output_count];

    ret = connx_Model_run(&model, input_count, inputs, &output_count, outputs);
    if(ret != CONNX_OK) {
		printf("Inference error: %d\n", ret);
		return;
	}

	for(uint32_t i = 0; i < output_count; i++) {
        connx_Tensor* output = outputs[i];
		connx_Tensor_dump(output);
	}

    fflush(stdout);
}
