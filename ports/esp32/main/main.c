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
#include "esp_camera.h"

#include <cam.h>
#include <connx/connx.h>

#define GET(row, col) (data + width * 3 * (row) + (col) * 3)
#define ROW_OFFSET 10
#define COL_OFFSET 300
#define ROW_BLOCK 25
#define COL_BLOCK 25

void app_main(void) {
	connx_init();

    // Parse connx model
    connx_Model model;
    int ret = connx_Model_init(&model);
    if(ret != 0) {
		connx_error("Cannot load model");
        return;
    }

    //void* buf = connx_load("test_data_set_0/input_0.data");
    //connx_Tensor* tensor = connx_Tensor_alloc_buffer(buf);
    //connx_unload(buf);
    //int32_t shape[] = {1, 1, 28, 28};
    //connx_Tensor* tensor = connx_Tensor_alloc(CONNX_FLOAT32, 4, shape);
    //float* array = (float*)tensor->buffer;
    //connx_Tensor_dump(tensor);

    ret = camera_init();
    printf("camera_init: %d\n", ret);

    while(true) {
        camera_fb_t* fb = camera_capture();
        printf("camera_capture: %u %u %p %u %d\n", fb->width, fb->height, fb->buf, fb->len, fb->format);

        uint8_t* buf = NULL;
        size_t buf_len = 0;
        bool converted = frame2bmp(fb, &buf, &buf_len);
        printf("converted: %d %p %u\n", converted, buf, buf_len);
        printf("%c%c %u %u\n", *(char*)(buf + 0), *(char*)(buf + 1), *(uint32_t*)(buf + 2), *(uint32_t*)(buf + 10));
        printf("%u %d %d %u %u\n", 
                *(uint32_t*)(buf + 14), // biSize
                *(int32_t*)(buf + 18), // biWidth
                *(int32_t*)(buf + 22), // biHeight
                *(uint32_t*)(buf + 30), // biCompression
                *(uint32_t*)(buf + 34)); // biSizeImage

        uint32_t width = fb->width;
        uint8_t* data = buf + *(uint32_t*)(buf + 10);

        int32_t shape[] = { 1, 1, 28, 28 };
        connx_Tensor* input = connx_Tensor_alloc(CONNX_FLOAT32, 4, shape);

        uint32_t input_count = 1;
        connx_Tensor* inputs[] = { input };

        uint32_t output_count = 16;
        connx_Tensor* outputs[output_count];

        float* input_flatten = (float*)input->buffer;
        float* p = input_flatten;

        float min = 255.0, max = 0.0;

        for(int row = 0; row < 28; row++) {
            for(int col = 27; col >= 0; col--) {
                float value = 255;

                for(int r = 0; r < ROW_BLOCK; r++) {
                    for(int c = 0; c < COL_BLOCK; c++) {
                        uint8_t* rgb = GET(ROW_OFFSET + row * ROW_BLOCK + r, COL_OFFSET + col * COL_BLOCK + c);
                        float tmp = 0;
                        for(int i = 0; i < 3; i++) {
                            tmp += (float)rgb[i];
                        }

                        if(tmp < value) {
                            value = tmp;
                        }
                    }
                }

                if(value != 0.0)
                    value /= 3;

                if(min > value)
                    min = value;

                if(max < value)
                    max = value;

                *p++ = value;
            }
        }

        if(converted) {
            free(buf);
        }

        camera_free(fb);

        float gap = max - min;
        if(gap == 0.0)
            gap = 0.001;
        p = input_flatten;
        for(int row = 0; row < 28; row++) {
            for(int col = 0; col < 28; col++) {
                *p = (1.0 - (*p - min) / gap) * 255.0;

                if(*p > 200)
                    printf("\e[31m%3d \e[m", (int)*p++);
                else
                    printf("%3d ", (int)*p++);
            }
            printf("\n");
        }

        ret = connx_Model_run(&model, input_count, inputs, &output_count, outputs);
        if(ret != CONNX_OK) {
            printf("Inference error: %d\n", ret);
            return;
        }

        for(uint32_t i = 0; i < output_count; i++) {
            connx_Tensor* output = outputs[i];
            connx_Tensor_dump(output);
        }

        int arg_max;
        max = 0.0;
        float* output_flatten = (float*)outputs[0]->buffer;
        for(int i = 0; i < 9; i++) {
            if(output_flatten[i] > max) {
                arg_max = i;
                max = output_flatten[i];
            }
        }

        printf("\e[32m%d %f\e[m\n", arg_max, max);

        fflush(stdout);
    }

}
