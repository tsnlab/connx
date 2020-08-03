#include <stdio.h>
#include <stdarg.h>
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
	if(file == NULL) {
		fprintf(stderr, "HAL ERROR: There is no such file: '%s'\n", path);
		return NULL;
	}

	fseek(file, 0L, SEEK_END);
	size_t size = ftell(file);
	fseek(file, 0L, SEEK_SET);

	void* buf = malloc(size);
	if(buf == NULL) {
		fprintf(stderr, "HAL ERROR: Cannot allocate memory: %ld bytes", size);
		fclose(file);
		return NULL;
	}

	void* p = buf;
	while(size > 0) {
		int len = fread(p, 1, size, file);
		if(len < 0) {
			fprintf(stderr, "HAL ERROR: Cannot read file: '%s'", path);
			fclose(file);
			return NULL;
		}

		p += len;
		size -= len;
	}
	fclose(file);

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

static bool ends_with(const char* text, char ch) {
	for(int i = 0; text[i] != '\0'; i++) {
		if(text[i] == ch && text[i + 1] == '\0') {
			return true;
		}
	}

	return false;
}

static void debug(__attribute__((unused)) connx_HAL* hal, const char* format, ...) {
	static bool is_head = true;

	va_list args;
	va_start(args, format);

	if(is_head) {
		fprintf(stdout, "DEBUG: ");

		for(uint32_t i = 0; i < hal->debug_tab; i++)
			fprintf(stdout, "\t");
	}

	vfprintf(stdout, format, args);

	va_end(args);

	is_head = ends_with(format, '\n');
}

static void info(__attribute__((unused)) connx_HAL* hal, const char* format, ...) {
	static bool is_head = true;

	va_list args;
	va_start(args, format);

	if(is_head) {
		fprintf(stdout, "INFO: ");

		for(uint32_t i = 0; i < hal->info_tab; i++)
			fprintf(stdout, "\t");
	}

	vfprintf(stdout, format, args);

	va_end(args);

	is_head = ends_with(format, '\n');
}

static void error(__attribute__((unused)) connx_HAL* hal, const char* format, ...) {
	static bool is_head = true;

	va_list args;
	va_start(args, format);

	if(is_head) {
		fprintf(stderr, "ERROR: ");

		for(uint32_t i = 0; i < hal->error_tab; i++)
			fprintf(stdout, "\t");
	}

	vfprintf(stderr, format, args);

	va_end(args);

	is_head = ends_with(format, '\n');
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
	hal->debug = debug;
	hal->info = info;
	hal->error = error;

	HALPriv* priv = (HALPriv*)hal->priv;
	snprintf(priv->path, 128, "%s", path);

	return hal;
}

void hal_delete(connx_HAL* hal) {
	free(hal);
}
