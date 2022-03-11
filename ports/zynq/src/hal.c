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
#include <inttypes.h>
#include <malloc.h>
#include <stdarg.h>
#include <string.h>

#include <connx/accel.h>
#include <connx/connx.h>
#include <connx/hal.h>
#include <connx/tensor.h>
#include <connx/hal_common.h>

#include "xparameters.h"	/* SDK generated parameters */
#include "xsdps.h"		/* SD device driver */
#include "xil_printf.h"
#include "ff.h"
#include "xil_cache.h"
#include "xplatform_info.h"

// Memory management
void* connx_alloc(uint32_t size) {
    return calloc(1, size);
}

void connx_free(void* ptr) {
    free(ptr);
}

void *_load(const char* name) {
	FRESULT Res;
    FIL file;
    FATFS fatfs;
	TCHAR *root = "0:/";

	Res = f_mount(&fatfs, root, 0);

	if (Res != FR_OK) {
		connx_error("%s %s\n", __func__, "f_mount fails\n");
		return XST_FAILURE;
	}

	Res = f_open(&file, name, FA_READ);
	if (Res) {
		connx_error("%s %s\n", __func__, "f_open fails\n");
		return XST_FAILURE;
	}

	Res = f_lseek(&file, f_size(&file));
	if (Res != FR_OK)
	{
		connx_error("%s %s\n", __func__, "f_lseek fails\n");
		return XST_FAILURE;
	}
	size_t size = f_tell(&file);

	Res = f_lseek(&file, 0);
	if (Res != FR_OK)
	{
		connx_error("%s %s\n", __func__, "f_lseek fails\n");
		return XST_FAILURE;
	}

	void *buf = malloc(size + 1);
	if (buf == NULL)
	{
		connx_error("%s %s\n", __func__, "HAL ERROR: Cannot allocate memory\n");
		f_close(&file);
		return XST_FAILURE;
	}

	void *p = buf;

	size_t remain = size;
	UINT br;

	while (remain > 0)
	{
		Res = f_read(&file, p, 1, &br);
		if (Res != FR_OK)
		{
			connx_error("%s %s\n", __func__, "f_read fails\n");
			f_close(&file);
			return XST_FAILURE;
		}
		p += br;
		remain -= br;
	}
	f_close(&file);

	((uint8_t*)buf)[size] = 0;

	return buf;
}

void* connx_load_model() {
    return _load("model.cnx");
}

void* connx_load_data(uint32_t graph_id, uint32_t id) {
    char name[16];
    snprintf(name, 16, "%u_%u.dat", graph_id, id);
    return _load(name);
}

void* connx_load_text(uint32_t graph_id) {
    char name[256];
    snprintf(name, 256, "%u.txt", graph_id);
    return _load(name);
}

static inline void _unload(void* buf) {
    free(buf);
}

void connx_unload_model(void* buf) {
    _unload(buf);
}

void connx_unload_data(void* buf) {
    _unload(buf);
}

void connx_unload_text(void* buf) {
    _unload(buf);
}

// Lock
void connx_Lock_init(connx_Lock* lock) {
    ;
}

void connx_Lock_destroy(connx_Lock* lock) {
    ;
}

void connx_Lock_lock(connx_Lock* lock) {
    ;
}

void connx_Lock_unlock(connx_Lock* lock) {
    ;
}

void connx_Thread_run_all(void* (*run)(void*), int32_t count, void* contexts, int32_t context_size) {
 #define BATCH_COUNT 16
    int32_t batch_count = BATCH_COUNT;
    if (batch_count > count)
        batch_count = count;
    for (int i = 0; i < batch_count; i++)
    {
        run(contexts);
        contexts += context_size;
    }
}

// error
void connx_debug(const char* format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(stderr, "DEBUG: ");
    vfprintf(stderr, format, args);

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

    for (int32_t i = 0; i < ndim; i++)
        fprintf(stderr, "%d ", index[i]);
    fprintf(stderr, "/ ");
    for (int32_t i = 0; i < ndim; i++)
        fprintf(stderr, "%d ", start[i]);
    fprintf(stderr, "/ ");
    for (int32_t i = 0; i < ndim; i++)
        fprintf(stderr, "%d ", stop[i]);
    fprintf(stderr, "/ ");
    for (int32_t i = 0; i < ndim; i++)
        fprintf(stderr, "%d ", step[i]);
    fprintf(stderr, "\n");
}

void connx_Tensor_dump(connx_Tensor* tensor) {
    int32_t unit = -1;
    int32_t unit2 = -1;

    if (tensor->ndim == 1) {
        unit = 8;
    } else if (tensor->ndim >= 1) {
        unit = tensor->shape[tensor->ndim - 1];

        if (tensor->ndim >= 2) {
            unit2 = unit * tensor->shape[tensor->ndim - 2];
        }
    }

    // New line by matrix
#define NEWLINE()              \
    if ((i + 1) % unit == 0)   \
        fprintf(stderr, "\n"); \
                               \
    if ((i + 1) % unit2 == 0)  \
        fprintf(stderr, "\n");

    int32_t total = connx_Int32_product(tensor->ndim, tensor->shape);
    connx_Tensor_dump_header(tensor);

    switch (tensor->dtype) {
    case CONNX_UINT8: {
        uint8_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
            fprintf(stderr, "%" PRIu8 " ", array[i]);
            NEWLINE()
        }
        fprintf(stderr, "\n");
        break;
    }
    case CONNX_INT8: {
        int8_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
            fprintf(stderr, "%" PRId8 " ", array[i]);
            NEWLINE()
        }
        fprintf(stderr, "\n");
        break;
    }
    case CONNX_UINT16: {
        uint16_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
            fprintf(stderr, "%" PRIu16 " ", array[i]);
            NEWLINE()
        }
        fprintf(stderr, "\n");
        break;
    }
    case CONNX_INT16: {
        int16_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
            fprintf(stderr, "%" PRId16 " ", array[i]);
            NEWLINE()
        }
        fprintf(stderr, "\n");
        break;
    }
    case CONNX_UINT32: {
        uint32_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
            fprintf(stderr, "%" PRIu32 " ", array[i]);
            NEWLINE()
        }
        fprintf(stderr, "\n");
        break;
    }
    case CONNX_INT32: {
        int32_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
            fprintf(stderr, "%" PRId32 " ", array[i]);
            NEWLINE()
        }
        fprintf(stderr, "\n");
        break;
    }
    case CONNX_UINT64: {
        uint64_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
            fprintf(stderr, "%" PRIu64 " ", array[i]);
            NEWLINE()
        }
        fprintf(stderr, "\n");
        break;
    }
    case CONNX_INT64: {
        int64_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
            fprintf(stderr, "%" PRId64 " ", array[i]);
            NEWLINE()
        }
        fprintf(stderr, "\n");
        break;
    }
    case CONNX_FLOAT16: {
        uint16_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
            fprintf(stderr, "%" PRIu16 " ", array[i]);
            NEWLINE()
        }
        fprintf(stderr, "\n");
        break;
    }
    case CONNX_FLOAT32: {
        float32_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
            fprintf(stderr, "%f ", array[i]);
            NEWLINE()
        }
        fprintf(stderr, "\n");
        break;
    }
    case CONNX_FLOAT64: {
        float64_t* array = tensor->buffer;
        for (int32_t i = 0; i < total; i++) {
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
        for (int32_t i = 0; i < total; i++) {
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

void connx_Tensor_dump_header(connx_Tensor* tensor) {
    fprintf(stderr, "tensor type(%d) < ", tensor->dtype);
    for (int32_t i = 0; i < tensor->ndim; i++) {
        fprintf(stderr, "%u ", tensor->shape[i]);
    }

    int32_t total = connx_Int32_product(tensor->ndim, tensor->shape);
    fprintf(stderr, "> = %u\n", total);
}
