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
#include <stdio.h>
#include <inttypes.h>
#include <malloc.h>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <connx/accel.h>
#include <connx/tensor.h>

#include <connx/hal.h>
#include <connx/types.h>

#include "esp_spiffs.h"
#include "esp_log.h"

// Lifecycle
void connx_init() {
    // SPIFFS
    esp_vfs_spiffs_conf_t conf = {
      .base_path = "/spiffs",
      .partition_label = NULL,
      .max_files = 5,
      .format_if_mount_failed = true
    };

    esp_err_t ret = esp_vfs_spiffs_register(&conf);

    if(ret != ESP_OK) {
        if(ret == ESP_FAIL) {
            ESP_LOGE("SPIFFS", "Failed to mount or format filesystem");
        } else if (ret == ESP_ERR_NOT_FOUND) {
            ESP_LOGE("SPIFFS", "Failed to find SPIFFS partition");
        } else {
            ESP_LOGE("SPIFFS", "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
        }

        return;
    }

    size_t total = 0, used = 0;
    ret = esp_spiffs_info(conf.partition_label, &total, &used);
    if(ret != ESP_OK) {
        ESP_LOGE("SPIFFS", "Failed to get SPIFFS partition information (%s)", esp_err_to_name(ret));
    } else {
        ESP_LOGE("SPIFFS", "Partition size: total: %d, used: %d", total, used);
    }
}

void connx_destroy() {
}

// Memory management
void* connx_alloc(uint32_t size) {
	if(size == 0) // ESP32 returns NULL when size is 0
		size = 1;

    return calloc(1, size);
}

void connx_free(void* ptr) {
    free(ptr);
}

// Model loader
void* connx_load(const char* name) {
	char path[128];
	snprintf(path, 128, "/spiffs/%s", name);

    FILE* file = fopen(path, "r");
    if(file == NULL) {
        fprintf(stderr, "HAL ERROR: There is no such file: '%s'\n", path);
        return NULL;
    }

    fseek(file, 0L, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0L, SEEK_SET);

    void* buf = malloc(size + 1); // including EOF
    if(buf == NULL) {
        fprintf(stderr, "HAL ERROR: Cannot allocate memory: %" PRIdPTR " bytes", size);
        fclose(file);
        return NULL;
    }

    void* p = buf;
    size_t remain = size;
    while(remain > 0) {
        int len = fread(p, 1, remain, file);
        if(len < 0) {
            fprintf(stderr, "HAL ERROR: Cannot read file: '%s'", path);
            fclose(file);
            return NULL;
        }

        p += len;
        remain -= len;
    }
    fclose(file);

    ((uint8_t*)buf)[size] = 0; // EOF

    return buf;
}

void connx_unload(__attribute__((unused)) void* buf) {
    free(buf);
}

// Lock
void connx_Lock_init(connx_Lock* lock) {
    pthread_mutex_init(lock, NULL);
}

void connx_Lock_destroy(connx_Lock* lock) {
    pthread_mutex_destroy(lock);
}

void connx_Lock_lock(connx_Lock* lock) {
    pthread_mutex_lock(lock);
}

void connx_Lock_unlock(connx_Lock* lock) {
    pthread_mutex_unlock(lock);
}

// Thread pool
uint32_t connx_Thread_alloc(uint32_t count, connx_Thread* threads) {
    return 0;
}

void connx_Thread_free(uint32_t count, connx_Thread* threads) {
}

void connx_Thread_join(uint32_t count, connx_Thread* threads) {
}

// error
void connx_debug(const char* format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(stdout, "DEBUG: ");
    vfprintf(stdout, format, args);

    va_end(args);
}

void connx_info(const char* format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(stdout, "INFO: ");
    vfprintf(stdout, format, args);

    va_end(args);
}

void connx_error(const char* format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, format, args);

    va_end(args);
}

// Below 5 lines are copied from tensor.c
#define ITER_NDIM(iter) (iter)
#define ITER_START(iter) (iter + 1)
#define ITER_STOP(iter) (iter + 1 + iter[0])
#define ITER_STEP(iter) (iter + 1 + iter[0] * 2)
#define ITER_INDEX(iter) (iter + 1 + iter[0] * 3)

void connx_Iterator_dump(int32_t* iterator) {
    int32_t ndim = *ITER_NDIM(iterator);
    int32_t* start = ITER_START(iterator);
    int32_t* stop = ITER_STOP(iterator);
    int32_t* step = ITER_STEP(iterator);
    int32_t* index = ITER_INDEX(iterator);

    for(int32_t i = 0; i < ndim; i++)
        printf("%d ", index[i]);
    printf("/ ");
    for(int32_t i = 0; i < ndim; i++)
        printf("%d ", start[i]);
    printf("/ ");
    for(int32_t i = 0; i < ndim; i++)
        printf("%d ", stop[i]);
    printf("/ ");
    for(int32_t i = 0; i < ndim; i++)
        printf("%d ", step[i]);
    printf("\n");
}

void connx_Tensor_dump(connx_Tensor* tensor) {
    int32_t unit = -1;
    int32_t unit2 = -1;

    if(tensor->ndim == 1) {
        unit = 8;
    } else if(tensor->ndim >= 1) {
        unit = tensor->shape[tensor->ndim - 1];

        if(tensor->ndim >= 2) {
            unit2 = unit * tensor->shape[tensor->ndim - 2];
        }
    }

    // New line by matrix
#define NEWLINE()              \
    if((i + 1) % unit == 0)    \
        fprintf(stderr, "\n"); \
                               \
    if((i + 1) % unit2 == 0)   \
        fprintf(stderr, "\n");

    fprintf(stderr, "tensor < ");
    for(int32_t i = 0; i < tensor->ndim; i++) {
        fprintf(stderr, "%u ", tensor->shape[i]);
    }

    int32_t total = connx_Int32_product(tensor->ndim, tensor->shape);
    fprintf(stderr, "> = %u\n", total);

    switch(tensor->dtype) {
        case CONNX_UINT8: {
            uint8_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRIu8 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_INT8: {
            int8_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRId8 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_UINT16: {
            uint16_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRIu16 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_INT16: {
            int16_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRId16 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_UINT32: {
            uint32_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRIu32 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_INT32: {
            int32_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRId32 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_UINT64: {
            uint64_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRIu64 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_INT64: {
            int64_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRId64 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_FLOAT16: {
            uint16_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRIu16 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_FLOAT32: {
            float32_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%f ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_FLOAT64: {
            float64_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%f ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_STRING:
            fprintf(stderr, "Not implemented yet\n");
            break;
        case CONNX_BOOL: {
            bool* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%s ", array[i] ? "true" : "false");
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case CONNX_COMPLEX64:
        case CONNX_COMPLEX128:
        case CONNX_UNDEFINED:
        default:
            fprintf(stderr, "Not implemented yet\n");
    }
}
