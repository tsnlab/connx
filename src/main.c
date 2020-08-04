#include <stdio.h>
#include <stdlib.h>
#define __USE_XOPEN_EXTENDED	// to use strdup
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include <getopt.h>
#include <sys/time.h>
#include <connx/connx.h>
#include <connx/dump.h>
#include "ver.h"

#define TITLE				"CONNX - C implementation of Open Neural Network Exchange Runtime"
#define COPYRIGHT_HOLDER	"Semih Kim"

static uint64_t get_us() {
	struct timeval  tv;
	gettimeofday(&tv, NULL);

	return (tv.tv_sec) * 1000000 + (tv.tv_usec);
}

/**
 * Consider there is enough space to enlarge
 */
static void pretty_number(char* buf) {
	int len = strlen(buf);
	int i = len % 3;
	if(i == 0)
		i = 3;
	for( ; i < len; i += 4, len++) {
		memmove(buf + i + 1, buf + i, len - i + 1);
		buf[i] = ',';
	}
}

static void print_notice() {
	printf("CONNX " CONNX_VERSION " Copyright (C) 2019-2020 " COPYRIGHT_HOLDER "\n\n");
    printf("This program comes with ABSOLUTELY NO WARRANTY.\n");
    printf("This is free software, and you are welcome to redistribute it\n");
    printf("under certain conditions; use -h option for details.\n\n");
}

static void print_copyright() {
	printf(TITLE "\n");
	printf("Copyright (C) 2019-2020 " COPYRIGHT_HOLDER "\n\n");

	printf("This program is free software: you can redistribute it and/or modify\n");
	printf("it under the terms of the GNU General Public License as published by\n");
	printf("the Free Software Foundation, either version 3 of the License, or\n");
	printf("(at your option) any later version.\n\n");

	printf("This program is distributed in the hope that it will be useful,\n");
	printf("but WITHOUT ANY WARRANTY; without even the implied warranty of\n");
	printf("MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n");
	printf("GNU General Public License for more details.\n\n");

	printf("You should have received a copy of the GNU General Public License\n");
	printf("along with this program.  If not, see <https://www.gnu.org/licenses/>.\n");
}

static void print_help() {
	print_copyright();

	printf("\nUsage:\n");
	printf("\tconnx [model] -i [input data] [-t [target data]] [-l [loop count]]\n\n");
	printf("Options:\n");
	printf("\t-i\tInput data\n");
	printf("\t-t\tTarget data\n");
    //printf("\t-c\tNumber of CPU cores to parallel processing\n");
	printf("\t-l\tLoop count (default is 1)\n");
	printf("\t-h\tDisplay this help message\n");
}

extern connx_HAL* hal_create(char* path);
extern void hal_delete(connx_HAL* hal);

int main(int argc, char** argv) {
	if(argc < 2) {
		print_notice();
		print_help();
		return 1;
	}

#define COUNT 16

	char* model = argv[1];
	uint32_t input_count = 0;
	char* input_names[COUNT];
	uint32_t target_count = 0;
	char* target_names[COUNT];
	uint32_t loop_count = 1;

	int option;
	while((option = getopt(argc, argv, "i:t:l:h")) != -1) {
		switch(option) {
			case 'i':
				if(input_count >= COUNT) {
					fprintf(stderr, "input count exceed: %u\n", COUNT);
					return 1;
				}
				input_names[input_count++] = strdup(optarg);
				break;
			case 't':
				if(target_count >= COUNT) {
					fprintf(stderr, "target count exceed: %u\n", COUNT);
					return 1;
				}
				target_names[target_count++] = strdup(optarg);
				break;
			case 'l':
				loop_count = strtoul(optarg, NULL, 0);
				break;
			case 'h':
				print_notice();
				print_help();
				return 0;
			default:
				print_notice();
				print_help();
				return 1;
		}
	}

	connx_HAL* hal = hal_create(model);

	// Make CONNX backend
	connx_Backend* backend = connx_Backend_create(hal);
	if(backend == NULL) {
		return -1;
	}

	// Load input tensors
	connx_Tensor* inputs[input_count];
	for(uint32_t i = 0; i < input_count; i++) {
		inputs[i] = connx_Backend_load_tensor(backend, input_names[i]);
		free(input_names[i]);
		if(inputs[i] == NULL) {
			fprintf(stderr, "Cannot load input tensor: '%s'\n", input_names[i]);
			return 1;
		}
	}

	// run
	uint32_t output_count = 16;
	connx_Tensor* outputs[output_count];
	bzero(outputs, sizeof(connx_Tensor*) * output_count);

	uint64_t time_start = get_us();

	for(uint32_t i = 0; i < loop_count; i++) {
		output_count = 16;

		for(uint32_t j = 0; j < output_count; j++) {
			if(outputs[j] != NULL) {
				connx_Tensor_delete(hal, outputs[j]);
			}
		}

		if(!connx_Backend_run(backend, &output_count, outputs, input_count, inputs)) {
			connx_Backend_delete(backend);
			hal_delete(hal);
			return -1;
		}
	}

	uint64_t time_end = get_us();

	char buf[32];
	sprintf(buf, "%" PRIu64, time_end - time_start);
	pretty_number(buf);
	printf("Time: %s us\n", buf);

	// Load target tensors
	connx_Tensor* targets[target_count];
	for(uint32_t i = 0; i < target_count; i++) {
		targets[i] = connx_Backend_load_tensor(backend, target_names[i]);
		free(target_names[i]);
		if(targets[i] == NULL) {
			fprintf(stderr, "Cannot load target tensor: '%s'\n", target_names[i]);
			return 1;
		}
	}

	bool is_succeed = true;

	uint32_t count = target_count > output_count ? target_count : output_count;
	for(uint32_t i = 0; i < count; i++) {
		if(i < target_count && i < output_count) {
			int32_t accuracy = connx_Tensor_accuracy(outputs[i], targets[i]);
			if(accuracy >= 0) {
				printf("output[%u] accuracy: 10^-%d\n", i, accuracy);
			} else {
				printf("output[%u] is incorrect\n", i);
				printf("output[%u]\n", i);
				connx_Tensor_dump(hal, outputs[i]);
				printf("target[%u]\n", i);
				connx_Tensor_dump(hal, targets[i]);
			}
		} else if(i >= target_count) {
			printf("Lack of output[%u]\n", i);
			is_succeed = false;
		} else {
			printf("Unexpected output[%u]\n", i);
			is_succeed = false;
		}
	}

	printf("Result: %s\n", is_succeed ? "SUCCEED" : "FAILED");

	// clean
	for(uint32_t i = 0; i < target_count; i++)
		connx_Tensor_delete(hal, targets[i]);

	for(uint32_t i = 0; i < output_count; i++)
		connx_Tensor_delete(hal, outputs[i]);

	for(uint32_t i = 0; i < input_count; i++)
		connx_Tensor_delete(hal, inputs[i]);

	connx_Backend_delete(backend);

	hal_delete(hal);

	return 0;
}
