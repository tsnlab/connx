#ifndef __CONNX_HAL_H__
#define __CONNX_HAL_H__

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#define CONNX_ALIGNMENT 16 // Data alignment
#define CONNX_ALIGN(offset) (((offset) + CONNX_ALIGNMENT - 1) & ~(CONNX_ALIGNMENT - 1))

// Lifecycle
void connx_init();
void connx_destroy();

// Memory management
void* connx_alloc(uint32_t size);
void connx_free(void* ptr);

// Model loader
void* connx_load(const char* name);
void connx_unload(void* buf);

// Tensor I/O
int32_t connx_read(void* buf, int32_t size);
int32_t connx_write(void* buf, int32_t size);

// Lock
typedef pthread_mutex_t connx_Lock;

void connx_Lock_init(connx_Lock* lock);
void connx_Lock_destroy(connx_Lock* lock);

void connx_Lock_lock(connx_Lock* lock);
void connx_Lock_unlock(connx_Lock* lock);

// Thread pool
typedef pthread_t connx_Thread;

uint32_t connx_Thread_alloc(uint32_t count, connx_Thread* threads);
void connx_Thread_free(uint32_t count, connx_Thread* threads);
void connx_Thread_join(uint32_t count, connx_Thread* threads);

// debugging message
struct _connx_Tensor;
void connx_debug(const char* format, ...);
void connx_info(const char* format, ...);
void connx_error(const char* format, ...);

void connx_Iterator_dump(int32_t* iterator);
void connx_Tensor_dump(struct _connx_Tensor* tensor);

#endif /* __CONNX_HAL_H__ */
