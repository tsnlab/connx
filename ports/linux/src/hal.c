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
#define _POSIX_C_SOURCE 199309L // clock_gettime
#include <inttypes.h>
#include <malloc.h>
#include <sched.h> // schd_yield
#include <stdarg.h>
#include <string.h>
#include <time.h>   // clock_gettime
#include <unistd.h> // sleep

#include <sys/stat.h>

#include <connx/accel.h>
#include <connx/hal.h>
#include <connx/tensor.h>

static char _model_path[128];
static FILE* _tensorin;
static FILE* _tensorout;

#define CONNX_WATCH_COUNT 20

static uint64_t _connx_watch_start[CONNX_WATCH_COUNT];
static uint64_t _connx_watch[CONNX_WATCH_COUNT];

#define MAX_THREAD_COUNT 16

struct thread {
    int32_t id;
    int status; // 0: started, 1: stopping, 2: stopped
    bool is_alloc;
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    void* (*run)(void*);
    void* context;
};

static connx_Lock _threads_lock;
static struct thread _threads[MAX_THREAD_COUNT];

static void* _run(void* context) {
    struct thread* thread = context;

    while (true) {
        pthread_mutex_lock(&thread->lock);

        if (thread->status != 0) {
            pthread_mutex_unlock(&thread->lock);
            break;
        }

        if (thread->run == NULL) {
            pthread_cond_wait(&thread->cond, &thread->lock);
        }

        pthread_mutex_unlock(&thread->lock);

        if (thread->run != NULL) {
            thread->run(thread->context);

            pthread_mutex_lock(&thread->lock);
            thread->run = NULL;
            thread->context = NULL;
            pthread_mutex_unlock(&thread->lock);
        }
    }

    pthread_mutex_lock(&thread->lock);
    thread->status = 2;
    pthread_mutex_unlock(&thread->lock);

    return NULL;
}

// Lifecycle
void connx_init() {
    pthread_mutex_init(&_threads_lock, NULL);

    for (int32_t i = 0; i < MAX_THREAD_COUNT; i++) {
        _threads[i].id = i;
        _threads[i].status = 0;
        _threads[i].is_alloc = false;
        pthread_mutex_init(&_threads[i].lock, NULL);
        pthread_cond_init(&_threads[i].cond, NULL);
        _threads[i].run = NULL;
        _threads[i].context = NULL;

        pthread_create(&_threads[i].thread, NULL, _run, &_threads[i]);
    }
}

void connx_destroy() {
    for (int32_t i = 0; i < MAX_THREAD_COUNT; i++) {
        struct thread* thread = &_threads[i];

        pthread_mutex_lock(&thread->lock);
        thread->status = 1;
        pthread_cond_signal(&thread->cond);
        pthread_mutex_unlock(&thread->lock);
    }

    for (int32_t i = 0; i < MAX_THREAD_COUNT; i++) {
        struct thread* thread = &_threads[i];

        while (true) {
            pthread_mutex_lock(&thread->lock);
            if (thread->status != 2) {
                pthread_cond_signal(&thread->cond);
                pthread_mutex_unlock(&thread->lock);

                sched_yield();
            } else {
                pthread_mutex_unlock(&thread->lock);
                break;
            }
        }

        pthread_cond_destroy(&_threads[i].cond);
        pthread_mutex_destroy(&_threads[i].lock);
    }

    if (_tensorin != NULL) {
        fclose(_tensorin);
    }

    if (_tensorout != NULL) {
        fclose(_tensorout);
    }
}

// Memory management
void* connx_alloc(uint32_t size) {
    return calloc(1, size);
}

void connx_free(void* ptr) {
    free(ptr);
}

// Model loader
int connx_set_model(const char* path) {
    snprintf(_model_path, 128, "%s", path);
    struct stat st;
    if (stat(_model_path, &st) == 0 && S_ISDIR(st.st_mode)) {
        return CONNX_OK;
    } else {
        connx_error("Model not found in path: '%s'\n", _model_path);
        return CONNX_RESOURCE_NOT_FOUND;
    }
}

int connx_set_tensorin(const char* path) {
    if (path == NULL && _tensorin != NULL) {
        fclose(_tensorin);
        _tensorin = NULL;
        return CONNX_OK;
    } else {
        if (strncmp("-", path, 2) == 0) {
            _tensorin = stdin;
        } else {
            _tensorin = fopen(path, "r");
        }

        if (_tensorin != NULL) {
            return CONNX_OK;
        } else {
            connx_error("Tensor input PIPE not found in path: '%s'\n", path);
            return CONNX_RESOURCE_NOT_FOUND;
        }
    }
}

int connx_set_tensorout(const char* path) {
    if (path == NULL && _tensorout != NULL) {
        fclose(_tensorout);
        _tensorout = NULL;
        return CONNX_OK;
    } else {
        if (strncmp("-", path, 2) == 0) {
            _tensorout = stdout;
        } else {
            _tensorout = fopen(path, "w");
        }

        if (_tensorout != NULL) {
            return CONNX_OK;
        } else {
            connx_error("Tensor output PIPE not found in path: '%s'\n", path);
            return CONNX_RESOURCE_NOT_FOUND;
        }
    }
}

void* connx_load(const char* name) {
    char path[256];
    snprintf(path, 256, "%s/%s", _model_path, name);

    FILE* file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "HAL ERROR: There is no such file: '%s'\n", path);
        return NULL;
    }

    fseek(file, 0L, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0L, SEEK_SET);

    void* buf = malloc(size + 1); // including EOF
    if (buf == NULL) {
        fprintf(stderr, "HAL ERROR: Cannot allocate memory: %" PRIdPTR " bytes", size);
        fclose(file);
        return NULL;
    }

    void* p = buf;
    size_t remain = size;
    while (remain > 0) {
        int len = fread(p, 1, remain, file);
        if (len < 0) {
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

// Tensor I/O
int32_t connx_read(void* buf, int32_t size) {
    FILE* file = _tensorin != NULL ? _tensorin : stdin;

    void* p = buf;
    size_t remain = size;
    while (remain > 0) {
        int len = fread(p, 1, remain, file);

        if (len < 0) {
            fprintf(stderr, "HAL ERROR: Cannot read input data");
            return -1;
        }

        p += len;
        remain -= len;
    }

    return size;
}

int32_t connx_write(void* buf, int32_t size) {
    FILE* file = _tensorout != NULL ? _tensorout : stdout;

    void* p = buf;
    size_t remain = size;
    while (remain > 0) {
        int len = fwrite(p, 1, remain, file);
        if (len < 0) {
            fprintf(stderr, "HAL ERROR: Cannot read input data");
            return -1;
        }

        p += len;
        remain -= len;
    }

    fflush(file);

    return size;
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
static int32_t _thread_alloc(int32_t count, uint32_t* thread_ids) {
    int32_t idx = 0;

    pthread_mutex_lock(&_threads_lock);

    for (int32_t i = 0; i < MAX_THREAD_COUNT && idx < count; i++) {
        if (!_threads[i].is_alloc) {
            thread_ids[idx++] = i;
            _threads[i].is_alloc = true;
        }
    }

    pthread_mutex_unlock(&_threads_lock);

    return idx;
}

static void _thread_free(int32_t count, uint32_t* thread_ids) {
    for (int32_t i = 0; i < count; i++) {
        struct thread* thread = &_threads[thread_ids[i]];
        while (true) {
            pthread_mutex_lock(&thread->lock);

            if (thread->run != NULL) {
                pthread_mutex_unlock(&thread->lock);
                sched_yield();
            } else {
                pthread_mutex_unlock(&thread->lock);
                break;
            }
        }
    }

    pthread_mutex_lock(&_threads_lock);

    for (int32_t i = 0; i < count; i++) {
        struct thread* thread = &_threads[thread_ids[i]];
        thread->is_alloc = false;
    }

    pthread_mutex_unlock(&_threads_lock);
}

void connx_Thread_run_all(void* (*run)(void*), int32_t count, void* contexts, int32_t context_size) {
#define BATCH_COUNT 16
    int32_t batch_count = BATCH_COUNT;
    if (batch_count > count)
        batch_count = count;

    uint32_t thread_ids[batch_count];
    int32_t thread_count = _thread_alloc(batch_count, thread_ids);

    for (int32_t work_idx = 0, thread_idx = 0; work_idx < count;) {
        int32_t thread_id = thread_ids[thread_idx];
        struct thread* thread = &_threads[thread_id];

        pthread_mutex_lock(&thread->lock);
        if (thread->run == NULL) {
            thread->run = run;
            thread->context = contexts;

            pthread_cond_signal(&thread->cond);

            pthread_mutex_unlock(&thread->lock);

            work_idx++;
            contexts += context_size;
        } else {
            pthread_mutex_unlock(&thread->lock);
        }

        if (++thread_idx >= thread_count) {
            thread_idx = 0;
            sched_yield();
        }
    }

    _thread_free(thread_count, thread_ids);
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

    fprintf(stderr, "tensor < ");
    for (int32_t i = 0; i < tensor->ndim; i++) {
        fprintf(stderr, "%u ", tensor->shape[i]);
    }

    int32_t total = connx_Int32_product(tensor->ndim, tensor->shape);
    fprintf(stderr, "> = %u\n", total);

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

uint64_t connx_time() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);

    return time.tv_sec * 1000000000ull + time.tv_nsec;
}

void connx_watch_start(int32_t idx) {
    if (idx >= 0 && idx < CONNX_WATCH_COUNT) {
        _connx_watch_start[idx] = connx_time();
    }
}

void connx_watch_stop(int32_t idx) {
    if (idx >= 0 && idx < CONNX_WATCH_COUNT) {
        _connx_watch[idx] += connx_time() - _connx_watch_start[idx];
    }
}

void connx_watch_dump() {
    bool has_dump = false;

    for (int32_t i = 0; i < CONNX_WATCH_COUNT; i++) {
        if (_connx_watch[i] != 0) {
            has_dump = true;
        }
    }

    if (has_dump) {
        for (int32_t i = 0; i < CONNX_WATCH_COUNT; i++) {
            fprintf(stderr, "Watch[%d] = %lu\n", i, _connx_watch[i]);
        }
    } else {
        fprintf(stderr, "Watch: nothing to dump\n");
    }
}
