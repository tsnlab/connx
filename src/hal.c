#include <malloc.h>
#include "hal.h"

#ifdef __linux__
#include <stdarg.h>
#endif /* __linux__ */

// Memory management
void* connx_alloc(uint32_t size) {
	return calloc(1, size);
}

void connx_free(void* ptr) {
	free(ptr);
}

// Model loader
void* connx_load(__attribute__((unused)) const char* name) {
    return NULL;
}

void connx_unload(__attribute__((unused)) void* buf) {
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

	fprintf(stdout, "ERROR: ");
	vfprintf(stdout, format, args);

	va_end(args);
}

