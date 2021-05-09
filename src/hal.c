#include "hal.h"

#ifdef __linux__
#include <inttypes.h>
#include <string.h>
#include <malloc.h>
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
static char _location[128];

void connx_set_location(const char* path) {
    snprintf(_location, 128, "%s", path);
}

void* connx_load(const char* name) {
#ifdef __linux__
    char path[256];
    snprintf(path, 256, "%s/%s", _location, name);

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
#else /* __linux__ */
    return NULL;
#endif
}

void connx_unload(__attribute__((unused)) void* buf) {
#ifdef __linux__
    free(buf);
#endif /* __linux__ */
}

// Tensor I/O
int32_t connx_read(void* buf, int32_t size) {
#ifdef __linux__
	void* p = buf;
    size_t remain = size;
	while(remain > 0) {
		int len = fread(p, 1, remain, stdin);
		if(len < 0) {
			fprintf(stderr, "HAL ERROR: Cannot read input data");
			return -1;
		}

		p += len;
		remain -= len;
	}

    return size;
#endif /* __linux __ */
}

int32_t connx_write(void* buf, int32_t size) {
#ifdef __linux__
	void* p = buf;
    size_t remain = size;
	while(remain > 0) {
		int len = fwrite(p, 1, remain, stdout);
		if(len < 0) {
			fprintf(stderr, "HAL ERROR: Cannot read input data");
			return -1;
		}

		p += len;
		remain -= len;
	}

    fflush(stdout);

    return size;
#endif /* __linux __ */
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

