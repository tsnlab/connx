#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/time.h>
#include <connx/connx.h>
#include "version.h"

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
    printf("under certain conditions; use -c option for details.\n\n");
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
	printf("Usage:\n");
	printf("\tconnx [onnx file] -i [input data] [-t [target data]] [-l [loop count]] [-e [tolerance number]] [-d]\n\n");
	printf("Options:\n");
	printf("\t-i\tInput data file (protocol buffer format)\n");
	printf("\t-t\tTarget data file (protocol buffer format)\n");
	printf("\t-o\tOutput data name\n");
	printf("\t-p\tDimensional parameter(key=value comma separated format,\n");
	printf("\t  \tkey _ is used for no named parameter)\n");
    printf("\t-c\tNumber of CPU cores to parallel processing\n");
	printf("\t-l\tLoop count (default is 1)\n");
    printf("\t-e\tTolerance number (default is 0.00001)\n");
	printf("\t-d\tDump variables\n");
	printf("\t-h\tDisplay this help message\n");
	printf("\t-c\tDisplay copyright\n");
}

int main(int argc, char** argv) {
	if(argc < 2) {
		print_notice();
		print_help();
		return 1;
	}

	char* fileOnnx = argv[1];
	uint32_t inputCount = 0;
	char* fileInputs[16] = { NULL, };
	connx_Tensor* inputs[16] = { 0, };
	uint32_t targetCount = 0;
	char* fileTargets[16] = { NULL, };
	connx_Tensor* targets[16] = { 0, };
	uint32_t outputCount = 0;
	char* outputs[16] = { NULL, };
	uint32_t loopCount = 1;
	bool isDebug = false;
	float tolerance = 0.00001;
	uint32_t parameterCount = 0;
	char** parameterNames = NULL;
	int64_t* parameterValues = NULL;
	int32_t coreCount = -1;

#define CLEAR()			\
	for(uint32_t i = 0; i < inputCount; i++) {	\
		if(fileInputs[i] != NULL) {				\
			connx_free(fileInputs[i]);			\
		}										\
		if(inputs[i] != NULL) {					\
			connx_Tensor_delete(inputs[i]);		\
		}										\
	}											\
												\
	for(uint32_t i = 0; i < targetCount; i++) {	\
		if(fileTargets[i] != NULL) {			\
			connx_free(fileTargets[i]);			\
		}										\
		if(targets[i] != NULL) {				\
			connx_Tensor_delete(targets[i]);	\
		}										\
	}											\
												\
	for(uint32_t i = 0; i < outputCount; i++) {	\
		if(outputs[i] != NULL) {			\
			connx_free(outputs[i]);			\
		}										\
	}											\
												\
	if(model != NULL) {							\
		connx_Model_delete(model);				\
	}
	
	int len;
	int option;
	while((option = getopt(argc, argv, "i:t:o:p:c:l:e:dhC")) != -1) {
		switch(option) {
			case 'i':
				len = strlen(optarg) + 1;
				fileInputs[inputCount] = connx_alloc(len);
				memcpy(fileInputs[inputCount], optarg, len);
				inputCount++;
				break;
			case 't':
				len = strlen(optarg) + 1;
				fileTargets[targetCount] = connx_alloc(len);
				memcpy(fileTargets[targetCount], optarg, len);
				targetCount++;
				break;
			case 'o':
				len = strlen(optarg) + 1;
				outputs[outputCount] = connx_alloc(len);
				memcpy(outputs[outputCount], optarg, len);
				outputCount++;
				break;
			case 'p':
				len = strlen(optarg) + 1;
				parameterCount = 1;
				for(int i = 0; i < len; i++) {
					if(optarg[i] == ',') {
						parameterCount++;
					}
				}

				parameterNames = connx_alloc(sizeof(char*) * parameterCount);
				parameterValues = connx_alloc(sizeof(int64_t) * parameterCount);

				int start = 0;
				for(uint32_t i = 0; i < parameterCount; i++) {
					// parse parameter name
					for(int j = start + 1; j < len; j++) {
						if(optarg[j] == '=') {
							int len2 = j - start;
							parameterNames[i] = connx_alloc(len2 + 1);
							memcpy(parameterNames[i], optarg + start, len2);
							parameterNames[i][len2] = '\0';
							start = j + 1;
							break;
						}
					}

					// parse parameter value
					for(int j = start + 1; j < len; j++) {
						if(optarg[j] == ',' || optarg[j] == '\0') {
							optarg[j] = '\0';
							parameterValues[i] = atoll(optarg + start);
							start = j + 1;
							break;
						}
					}
				}
				break;
			case 'c':
				coreCount = (int32_t)atoi(optarg);
				break;
			case 'l':
				loopCount = (uint32_t)atoi(optarg);
				break;
			case 'e':
				tolerance = atof(optarg);
				break;
			case 'd':
				isDebug = true;
				break;
			case 'h':
				print_notice();
				print_help();
				return 0;
			case 'C':
				print_copyright();
				return 0;
			default:
				print_notice();
				print_help();
				return 1;
		}
	}

	print_notice();

	connx_init(coreCount);

	connx_Model* model = connx_Model_create_from_file(fileOnnx);
	if(model == NULL) {
		fprintf(stderr, "Cannot read model file: %s\n", fileOnnx);
		CLEAR();
		return 1;
	}

	// load inputs
	for(uint32_t i = 0; i < inputCount; i++) {
		inputs[i] = connx_Tensor_create_from_file(fileInputs[i]);
		if(inputs[i] == NULL) {
			fprintf(stderr, "Cannot read input file: %s\n", fileInputs[i]);
			CLEAR();
			return 1;
		}

		if(inputs[i]->name == NULL && model->graph->n_input > i) {
			inputs[i]->name = model->graph->input[i]->name;
		}
	}

	if(isDebug) {
		printf("* inputs\n");

		for(uint32_t j = 0; j < inputCount; j++) {
			connx_Tensor* tensor = inputs[j];

			printf("\t%s: (", tensor->name);
			for(uint32_t i = 0; i < tensor->dimension; i++) {
				printf("%" PRIu32, tensor->lengths[i]);
				if(i + 1 < tensor->dimension) {
					printf(", ");
				}
			}
			printf(")\n");
		}
	}

	// load targets
	for(uint32_t i = 0; i < targetCount; i++) {
		targets[i] = connx_Tensor_create_from_file(fileTargets[i]);
		if(targets[i] == NULL) {
			fprintf(stderr, "Cannot read target file: %s\n", fileTargets[i]);
			CLEAR();
			return 1;
		}

		if(targets[i]->name == NULL && model->graph->n_output > i) {
			targets[i]->name = model->graph->output[i]->name;
		}
	}

	if(isDebug) {
		printf("* targets\n");

		for(uint32_t j = 0; j < targetCount; j++) {
			connx_Tensor* tensor = targets[j];

			printf("\t%s: (", tensor->name);
			for(uint32_t i = 0; i < tensor->dimension; i++) {
				printf("%" PRIu32, tensor->lengths[i]);
				if(i + 1 < tensor->dimension) {
					printf(", ");
				}
			}
			printf(")\n");
		}
	}

	if(isDebug) {
		printf("* producer_name : %s\n", model->producer_name);
		printf("* producer_version : %s\n", model->producer_version);
		printf("* graph.name: %s\n", model->graph->name);
		printf("* graph.input\n");
		for(uint32_t i = 0; i < model->graph->n_input; i++) {
			printf("\t%s\n", model->graph->input[i]->name);
		}
		printf("* graph.output\n");
		for(uint32_t i = 0; i < model->graph->n_output; i++) {
			printf("\t%s\n", model->graph->output[i]->name);
		}

		connx_Model_dump(model);
	}

	connx_Runtime* runtime = connx_Runtime_create(model);
	if(runtime == NULL) {
		CLEAR();
		return 1;
	}

	runtime->parameterCount = parameterCount;
	runtime->parameterNames = parameterNames;
	runtime->parameterValues = parameterValues;

	if(!connx_Runtime_init(runtime)) {
		CLEAR();
		return 2;
	}

	uint64_t time_start = get_us();
	for(uint32_t i = 0; i < loopCount; i++) {
		connx_Runtime_run(runtime, inputCount, (connx_Value**)inputs);
	}
	uint64_t time_end = get_us();

	char buf[32];
	sprintf(buf, "%" PRIu64, time_end - time_start);
	pretty_number(buf);
	printf("Time: %s us\n", buf);

	if(outputCount > 0) {
		for(uint32_t i = 0; i < outputCount; i++) {
			connx_Value* output = connx_Runtime_getVariable(runtime, outputs[i]);

			if(output == NULL) {
				connx_exception("Output %s not found.\n", outputs[i]);
			} else if(output->type == connx_DataType_TENSOR) {
				connx_Tensor_dump((connx_Tensor*)output);
			} else {
				printf("Cannot dump value: %s\n", outputs[i]);
			}
		}
	} else {
		for(uint32_t i = 0; i < targetCount; i++) {
			connx_Tensor* target = targets[i];
			connx_Value* output = connx_Runtime_getVariable(runtime, target->name);

			if(output == NULL) {
				connx_exception("Target %s not found.\n", target->name);
			} else if(output->type != connx_DataType_TENSOR) {
				char buf[32];
				connx_DataType_toString(output->type, 32, buf);
				connx_exception("Target %s type not matching: %s.\n", target->name, buf);
			} else {
				connx_Tensor* tensor = (connx_Tensor*)output;

				if(connx_Tensor_equals(tensor, target)) {
					printf("Target matched\n");
				} else if(connx_Tensor_isNearlyEquals(tensor, target, tolerance)) {
					printf("Target nearly matched: epsilon is %f\n", tolerance);
				} else {
					printf("Target not matched\n");
					target->name = "target";
					connx_Tensor_dump_compare(tensor, target, tolerance);
				}
			}
		}
	}

	connx_Runtime_delete(runtime);
	CLEAR();
	connx_finalize();

	return 0;
}
