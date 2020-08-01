#include <stdio.h>
#include <connx/connx.h>

extern connx_HAL* hal_create(char* path);

int main(__attribute__((unused)) int argc, __attribute__((unused)) char** argv) {
	connx_HAL* hal = hal_create("out");
	connx_Backend* backend = connx_Backend_create(hal);
	if(backend == NULL) {
		return -1;
	}

	connx_Backend_delete(backend);

	return 0;
}
