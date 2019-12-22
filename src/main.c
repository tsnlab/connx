#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/time.h>
#include <connx/connx.h>

#define TITLE				"CONNX - C implementation of Open Neural Network Exchange Runtime"
#define COPYRIGHT_HOLDER	"Semih Kim"

uint64_t get_us() {
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
	printf("CONNX  Copyright (C) 2019  " COPYRIGHT_HOLDER "\n\n");
    printf("This program comes with ABSOLUTELY NO WARRANTY.\n");
    printf("This is free software, and you are welcome to redistribute it\n");
    printf("under certain conditions; use -c option for details.\n\n");
}

static void print_copyright() {
	printf(TITLE "\n");
	printf("Copyright (C) 2019  " COPYRIGHT_HOLDER "\n\n");

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
	printf("Usage:\n");
	printf("\tconnx [onnx file] -i [input data] [-t [target data]] [-l [loop count]] [-d]\n\n");
	printf("Options:\n");
	printf("\t-i	Input data file (protocol buffer format)\n");
	printf("\t-t	Target data file (protocol buffer format)\n");
	printf("\t-l	Loop count (default is 1)\n");
	printf("\t-d	Dump variables\n");
	printf("\t-h	Display this help message\n");
	printf("\t-v	Display this application version\n");
	printf("\t-c	Display copyright\n");
}

static void print_version() {
	printf("CONNX ver 0.0.0\n");
}

int main(int argc, char** argv) {
	if(argc < 2) {
		print_notice();
		print_help();
		return 1;
	}

	char* fileOnnx = argv[1];
	char* fileInput = NULL;
	char* fileTarget = NULL;
	uint32_t loopCount = 1;
	bool isDebug = false;

	int option;
	while((option = getopt(argc, argv, "i:t:l:dhvc")) != -1) {
		switch(option) {
			case 'i':
				fileInput = optarg;
				break;
			case 't':
				fileTarget = optarg;
				break;
			case 'l':
				loopCount = (uint32_t)atoi(optarg);
				break;
			case 'd':
				isDebug = true;
				break;
			case 'h':
				print_notice();
				print_help();
				return 0;
			case 'v':
				print_notice();
				print_version();
				return 0;
			case 'c':
				print_copyright();
				return 0;
			default:
				print_notice();
				print_help();
				return 1;
		}
	}

	print_notice();

	if(fileInput == NULL) {
		fprintf(stderr, "You must specify input data.\n");
		print_help();
		return 1;
	}

	connx_init();
	connx_Tensor* input = connx_Tensor_create_from_file(fileInput);
	if(isDebug) {
		printf("* input data\n");
		connx_Tensor_dump(input);
	}

	connx_Model* model = connx_Model_create_from_file(fileOnnx);
	if(isDebug) {
		printf("* model\n");
		connx_Model_dump(model);
	}

	if(isDebug) {
		printf("* operators\n");
		connx_Operator_dump();
	}

	connx_Runtime* runtime = connx_Runtime_create(model);
	connx_Value* result;
	uint64_t time_start = get_us();
	for(uint32_t i = 0; i < loopCount; i++) {
		result = connx_Runtime_run(runtime, (connx_Value*)input);
	}
	uint64_t time_end = get_us();

	char buf[32];
	sprintf(buf, "%lu", time_end - time_start);
	pretty_number(buf);
	printf("Time: %s us\n", buf);

	if(result != NULL && result->type == connx_DataType_TENSOR) {
		connx_Tensor* tensor = (connx_Tensor*)result;
		connx_Tensor_dump(tensor);

		if(fileTarget != NULL) {
			connx_Tensor* target = connx_Tensor_create_from_file(fileTarget);
			if(connx_Tensor_equals(tensor, target)) {
				printf("Target matched\n");
			} else {
				printf("Target not matched\n");
				target->name = "target";
				connx_Tensor_dump(target);
			}
			connx_Tensor_delete(target);
		}
	}

	connx_Tensor_delete(input);
	connx_Runtime_delete(runtime);
	connx_Model_delete(model);

	return 0;
}
