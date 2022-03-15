/*
 *  CONNX, C implementation of Open Neural Network Exchange Runtime
 *  Copyright (C) 2019-2021 TSN Lab, Inc.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <cam.h>
#include <stdio.h>
#include <string.h>

#include <sys/unistd.h>

#include <connx/connx.h>
#include <connx/tensor.h>

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

uint64_t get_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000ULL + (tv.tv_usec / 1000ULL));
}

static void min_pool(camera_fb_t* fb, int32_t rows, int32_t cols, uint8_t* red, uint8_t* green, uint8_t* blue) {
#define GET(row, col) (data + width * 3 * ((row)-1) - ((col) + 1) * 3)
    int32_t height = fb->height;
    int32_t width = fb->width;

    int32_t block1 = height / rows;
    int32_t block2 = width / cols;
    int32_t block = block1 < block2 ? block1 : block2;

    int32_t row_offset = (height - rows * block) / 2;
    int32_t col_offset = (width - cols * block) / 2;

    uint8_t* rp = red;
    uint8_t* gp = green;
    uint8_t* bp = blue;

    uint8_t* buf = NULL;
    size_t buf_len = 0;
    bool converted = frame2bmp(fb, &buf, &buf_len);
    if (!converted) {
        ESP_LOGE("Main", "Cannot convert the frame to bmp");
        return;
    }

    uint8_t* data = buf + *(uint32_t*)(buf + 10);

    for (int32_t row = 0; row < rows; row++) {
        for (int32_t col = 0; col < cols; col++) {
            uint8_t* rgb = GET(row_offset + row * block, col_offset + col * block);
            uint8_t r = rgb[0], g = rgb[1], b = rgb[2];

            for (int32_t r2 = 0; r2 < block; r2++) {
                for (int32_t c2 = 0; c2 < block; c2++) {
                    rgb = GET(row_offset + row * block + r2, col_offset + col * block + c2);

                    if (rgb[0] < r) {
                        r = rgb[0];
                    }

                    if (rgb[1] < g) {
                        g = rgb[1];
                    }

                    if (rgb[2] < b) {
                        b = rgb[2];
                    }
                }
            }

            *rp++ = r;
            *gp++ = g;
            *bp++ = b;
        }
    }

    free(buf);
}

static void grayscale(int32_t length, uint8_t* gray, uint8_t* red, uint8_t* green, uint8_t* blue) {
    uint8_t* op = gray;
    uint8_t* rp = red;
    uint8_t* gp = green;
    uint8_t* bp = blue;

    for (int32_t i = 0; i < length; i++) {
        *op++ = (*rp++ + *gp++ + *bp++) / 3.0;
    }
}

static void invert_regularize(int32_t length, connx_Tensor* tensor, uint8_t* array) {
    uint8_t min = array[0];
    uint8_t max = array[0];
    for (int32_t i = 1; i < length; i++) {
        if (array[i] < min) {
            min = array[i];
        }

        if (array[i] > max) {
            max = array[i];
        }
    }

    if (max != min) {
        float* p = (float*)tensor->buffer;
        float scale = 255.0 / (max - min);

        for (int32_t i = 0; i < length; i++) {
            p[i] = 255.0 - (array[i] - min) * scale;
        }
    }
}

portMUX_TYPE mutex = portMUX_INITIALIZER_UNLOCKED;
int status = 0;

uint8_t* red;
uint8_t* green;
uint8_t* blue;
uint8_t* gray;
connx_Tensor* input;

void core0(void* args) {
    int ret = camera_init();
    if (ret != 0) {
        ESP_LOGE("CAM", "Cannot initialize the camera");
        return;
    }

    printf("Core0 : %s\n", (char*)args);
    while (true) {
        uint64_t start_time = get_ms();
        camera_fb_t* fb = camera_capture();
        uint64_t end_time = get_ms();
        printf("Capture:       \e[33m%llu ms\e[m\n", end_time - start_time);

        start_time = end_time;
        min_pool(fb, 28, 28, red, green, blue);
        camera_free(fb);
        grayscale(28 * 28, gray, red, green, blue);
        invert_regularize(28 * 28, input, gray);

        // Wait for core1 to copy gray to tensor
        while (true) {
            portENTER_CRITICAL(&mutex);
            if (status == 0) {
                portEXIT_CRITICAL(&mutex);
                break;
            }
            portEXIT_CRITICAL(&mutex);
            vTaskDelay(1);
        }

        end_time = get_ms();
        printf("Preprocessing: \e[33m%llu ms\e[m\n", end_time - start_time);

        portENTER_CRITICAL(&mutex);
        status = 1;
        portEXIT_CRITICAL(&mutex);

        start_time = end_time;
        float* p = (float*)input->buffer;
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                if (*p > 200)
                    printf("\e[31m%3d \e[m", (int)*p++);
                else
                    printf("%3d ", (int)*p++);
            }
            printf("\n");
        }
        end_time = get_ms();
        printf("Dump image:    \e[33m%llu ms\e[m\n", end_time - start_time);

        vTaskDelay(1);
    }
}

void core1(void* args) {
    connx_init();

    int32_t shape[] = {1, 1, 28, 28};
    input = connx_Tensor_alloc(CONNX_FLOAT32, 4, shape);

    uint32_t input_count = 1;
    connx_Tensor* inputs[] = {input};

    uint32_t output_count = 16;
    connx_Tensor* outputs[output_count];

    // Parse connx model
    connx_Model model;
    int ret = connx_Model_init(&model);
    if (ret != 0) {
        ESP_LOGE("CONNX", "Cannot load model");
        return;
    }

    while (true) {
        // Wait for core0 to preprocessing
        while (true) {
            portENTER_CRITICAL(&mutex);
            if (status == 1) {
                portEXIT_CRITICAL(&mutex);
                break;
            }
            portEXIT_CRITICAL(&mutex);
            vTaskDelay(1);
        }

        uint64_t start_time = get_ms();
        connx_Tensor_ref(input);
        ret = connx_Model_run(&model, input_count, inputs, &output_count, outputs);
        if (ret != CONNX_OK) {
            printf("Inference error: %d\n", ret);
            return;
        }
        uint64_t end_time = get_ms();
        printf("Inference:     \e[33m%llu ms\e[m\n", end_time - start_time);

        int arg_max;
        float max = 0.0;
        float* output_flatten = (float*)outputs[0]->buffer;
        for (int i = 0; i < 9; i++) {
            if (output_flatten[i] > max) {
                arg_max = i;
                max = output_flatten[i];
            }
        }
        printf("\e[32m%d %f\e[m\n", arg_max, max);

        start_time = end_time;
        for (uint32_t i = 0; i < output_count; i++) {
            connx_Tensor* output = outputs[i];
            connx_Tensor_dump(output);
        }
        end_time = get_ms();
        printf("Dump tensor:   \e[33m%llu ms\e[m\n", end_time - start_time);

        portENTER_CRITICAL(&mutex);
        status = 0;
        portEXIT_CRITICAL(&mutex);

        vTaskDelay(1);
    }
}

#define STACK_SIZE_0 4096
#define STACK_SIZE_1 4096
StackType_t stack_0[STACK_SIZE_0];
StackType_t stack_1[STACK_SIZE_1];
StaticTask_t task_0;
StaticTask_t task_1;

void app_main(void) {
    red = malloc(28 * 28);
    green = malloc(28 * 28);
    blue = malloc(28 * 28);
    gray = malloc(28 * 28);

    xTaskCreateStaticPinnedToCore(core0, "I/O", STACK_SIZE_0, "Core #0", 1, stack_0, &task_0, 0);
    xTaskCreateStaticPinnedToCore(core1, "Inference", STACK_SIZE_1, "Core #1", 1, stack_1, &task_1, 1);

    while (true) {
        vTaskDelay(1);
    }
}
