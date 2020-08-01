#include <stdio.h>
#include <dirent.h> 
#include <malloc.h>
#include <connx/connx.h>

typedef struct {
	char path[128];
} HALPriv;

// Memory management
static void* mem_alloc(__attribute__((unused)) connx_HAL* hal, size_t size) {
	return calloc(1, size);
}

static void mem_free(__attribute__((unused)) connx_HAL* hal, void* ptr) {
	free(ptr);
}

// Model loader
static void* load(connx_HAL* hal, const char* name) {
	HALPriv* priv = (HALPriv*)hal->priv;
	char path[256];
	snprintf(path, 256, "%s/%s", priv->path, name);

	FILE* file = fopen(path, "r");
	if(file == NULL)
		return NULL;

	fseek(file, 0L, SEEK_END);
	long len = ftell(file) + 1;
	fseek(file, 0L, SEEK_SET);

	void* buf = malloc(len);
	if(buf == NULL) {
		fclose(file);
		return NULL;
	}

	size_t len2 = fread(buf, 1, len, file);
	fclose(file);
	((uint8_t*)buf)[len - 1] = 0;

	if((long)len2 + 1 != len) {
		free(buf);
		fclose(file);
		return NULL;
	}

	return buf;
}

static void unload(__attribute__((unused)) connx_HAL* hal, void* buf) {
	free(buf);
}

// Thread pool
static connx_Thread* alloc_threads(connx_HAL* hal, uint32_t count) {
	return NULL;
}

static void free_thread(connx_HAL* hal, connx_Thread* thread) {
}

static connx_Thread* join(connx_HAL* hal, connx_Thread* thread) {
	return NULL;
}

static void info(connx_HAL* hal, const char* msg) {
	fprintf(stdout, "INFO: %s\n", msg);
}

static void error(connx_HAL* hal, const char* msg) {
	fprintf(stderr, "ERROR: %s\n", msg);
}

connx_HAL* hal_create(char* path) {
	connx_HAL* hal = calloc(1, sizeof(connx_HAL) + sizeof(HALPriv));
	hal->alloc = mem_alloc;
	hal->free = mem_free;
	hal->load = load;
	hal->unload = unload;
	hal->alloc_threads = alloc_threads;
	hal->free_thread = free_thread;
	hal->join = join;
	hal->info = info;
	hal->error = error;

	HALPriv* priv = (HALPriv*)hal->priv;
	snprintf(priv->path, 128, "%s", path);

	return hal;
}
