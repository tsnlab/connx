#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <malloc.h>
#include <connx/connx.h>

#define EPSILON 0.00001
#define GREEN "\033[0;32m"
#define RED "\033[0;31m"
#define END "\033[0m"

static char** testcases;

static bool read_filelist() {
	DIR* dir = opendir("testcase");
	if(dir == NULL) {
		fprintf(stderr, "Cannot open testcase directory\n");
		return false;
	}

	// get count
	uint32_t count = 1;	// last entity will be NULL
	struct dirent *entity;
	while((entity = readdir(dir)) != NULL) {
		count++;
	}

	testcases = calloc(1, sizeof(char*) * count);

	uint32_t idx = 0;
	rewinddir(dir);
	while((entity = readdir(dir)) != NULL) {
		int len = strlen(entity->d_name);
		testcases[idx] = calloc(1, len + 1);
		memcpy(testcases[idx], entity->d_name, len);
		idx++;
	}
	testcases[idx] = NULL;

	closedir(dir);

	return true;
}

static char* buf;
static int idx;
static int len;

static bool read_testcase(const char* path) {
	FILE* file = fopen(path, "r");
	if(file == NULL) {
		fprintf(stderr, "Cannot read file: %s\n", path);
		return false;
	}

	fseek(file, 0, SEEK_END);
	len = ftell(file);
	fseek(file, 0, SEEK_SET);

	buf = malloc(len + 1);
	if(buf == NULL) {
		fprintf(stderr, "Not enough memory\n");
		return false;
	}

	size_t len2 = fread(buf, 1, len, file);
	if((size_t)len != len2) {
		fprintf(stderr, "Cannot read file fully: %s\n", path);
		return false;
	}
	buf[len] = 0;

	fclose(file);

	return true;
}

uint32_t stackIdx;
uintptr_t stack[16];

char* readline() {
	if(idx >= len)
		return NULL;

	char* line = buf + idx;

	while(buf[idx] != '\n')
		idx++;
	buf[idx++] = '\0';

	return line;
}

char* parse_header(char *YYCURSOR, char** _id, char** _kind, char** _type);
char* parse_int(char *YYCURSOR, char** _number);
char* parse_float(char *YYCURSOR, char** _number);
char* parse_id(char *YYCURSOR, char** _id);

static bool exec_testcase(connx_Operator* op) {
	char* line = readline();
	printf("\t* Test case: %s ", line);
	fflush(stdout);

	// Read variables
	stackIdx = 1;
	line = readline();
	while(line != NULL) {
		while(strchr(line, '=') == NULL)
			line = readline();

		char* id = NULL;
		char* kind = NULL;
		char* type = NULL;
		line = parse_header(line, &id, &kind, &type);

		if(strcmp(kind, "tensor") == 0) {
			connx_DataType elemType;
			if(strcmp("float32", type) == 0) {
				elemType = connx_DataType_FLOAT32;
			} else if(strcmp("int64", type) == 0) {
				elemType = connx_DataType_INT64;
			} else {
				fprintf(stderr, "Not supported type: %s\n", type);
				abort();
			}

			uint32_t dimension = 0;
			uint32_t lengths[16];
			while(line != NULL) {
				char* num = NULL;
				line = parse_int(line, &num);
				if(num != NULL) {
					lengths[dimension++] = (uint32_t)strtoul(num, NULL, 10);
				}
			}

			connx_Tensor* tensor = connx_Tensor_create2(elemType, dimension, lengths);
			tensor->name = id;

			if(elemType == connx_DataType_FLOAT32) {
				uint32_t i = 0;
				float* base = (float*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_float(line, &num);
						if(num != NULL) {
							base[i++] = strtof(num, NULL);
						}
					}

					line = readline();
				}
			} else if(elemType == connx_DataType_INT64) {
				uint32_t i = 0;
				int64_t* base = (int64_t*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_int(line, &num);
						if(num != NULL) {
							base[i++] = strtoll(num, NULL, 10);
						}
					}

					line = readline();
				}
			} else {
				abort();
			}

			stack[stackIdx++] = (uintptr_t)tensor;
		} else if(strcmp(kind, "attribute") == 0) {
			if(strcmp("float", type) == 0) {
				char* num = NULL;
				parse_float(line, &num);

				stack[stackIdx++] = (uintptr_t)connx_Attribute_create_float(num != NULL ? strtof(num, NULL) : 0);
				line = readline();
			} else if(strcmp("floats", type) == 0) {
				float base[32];
				int count = 0;

				while(line != NULL) {
					char* num = NULL;
					line = parse_float(line, &num);
					if(num != NULL) {
						base[count++] = strtof(num, NULL);
					}
				}

				stack[stackIdx++] = (uintptr_t)connx_Attribute_create_floats(count, base);
				line = readline();
			} else if(strcmp("int", type) == 0) {
				char* num = NULL;
				parse_int(line, &num);

				stack[stackIdx++] = (uintptr_t)connx_Attribute_create_int(num != NULL ? strtol(num, NULL, 10) : 0);
				line = readline();
			} else if(strcmp("ints", type) == 0) {
				int64_t base[32];
				int count = 0;

				while(line != NULL) {
					char* num = NULL;
					line = parse_int(line, &num);
					if(num != NULL) {
						base[count++] = strtoll(num, NULL, 10);
					}
				}

				stack[stackIdx++] = (uintptr_t)connx_Attribute_create_ints(count, base);
				line = readline();
			} else if(strcmp("string", type) == 0) {
				char* id = NULL;
				parse_id(line, &id);

				stack[stackIdx++] = (uintptr_t)connx_Attribute_create_string(id != NULL ? id : "");
				line = readline();
			} else if(strcmp("strings", type) == 0) {
				char* id = NULL;
				char* base[32];
				int count = 0;

				while(line != NULL) {
					line = parse_id(line, &id);
					if(id != NULL) {
						int len = strlen(id + 1);
						base[count] = connx_alloc(len);
						memcpy(base[count], id, len);
						count++;
					}
				}

				stack[stackIdx++] = (uintptr_t)connx_Attribute_create_strings(count, base);
				line = readline();
			} else {
				fprintf(stderr, "Not supported type: %s\n", type);
				abort();
			}
		} else {
			fprintf(stderr, "Illegal variable kind: %s\n", kind);
			abort();
		}
	}

	stack[0] = stackIdx - 2;

	if(stack[0] != op->outputCount + op->inputCount + op->attributeCount) {
		printf(RED "\tSTACK COUNT MISMATCH: expected: %u but %lu\n", op->outputCount + op->inputCount + op->attributeCount, stack[0]);
		return false;
	}

	if(!op->resolve(stack)) {
		printf(RED "\tRESOLVE FAILED: %s\n" END, connx_exception_message());
		return false;
	}

	if(!op->exec(stack)) {
		printf(RED "\tEXECUTION FAILED: %s\n" END, connx_exception_message());
		return false;
	}

	if(connx_Tensor_isNearlyEquals((connx_Tensor*)stack[1], (connx_Tensor*)stack[stackIdx - 1], EPSILON)) {
		printf(GREEN "\tPASS\n" END);
	} else {
		printf(RED "\tFAILED\n" END);
		connx_Tensor_dump_compare((connx_Tensor*)stack[1], (connx_Tensor*)stack[stackIdx - 1], EPSILON);
	}

	// clean up
	// delete outputs
	uint32_t stackIdx2 = 1;
	for(uint32_t i = 0; i < op->outputCount; i++, stackIdx2++) {
		if(stack[stackIdx2] != 0) {
			connx_Tensor* tensor = (connx_Tensor*)stack[stackIdx2];
			connx_Tensor_delete(tensor);
		}
	}

	// delete inputs
	for(uint32_t i = 0; i < op->inputCount; i++, stackIdx2++) {
		if(stack[stackIdx2] != 0) {
			connx_Tensor* tensor = (connx_Tensor*)stack[stackIdx2];
			connx_Tensor_delete(tensor);
		}
	}

	// delete attributes
	for(uint32_t i = 0; i < op->attributeCount; i++, stackIdx2++) {
		if(stack[stackIdx2] != 0) {
			connx_Attribute* attr = (connx_Attribute*)stack[stackIdx2];
			connx_Attribute_delete(attr);
		}
	}

	// delete _result
	if(stack[stackIdx2] != 0) {
		connx_Tensor* tensor = (connx_Tensor*)stack[stackIdx2];
		connx_Tensor_delete(tensor);
	}

	free(buf);
	buf = NULL;
	idx = 0;
	len = 0;

	return true;
}

int main(__attribute((unused)) int argc, __attribute((unused)) char** argv) {
	connx_init();

	read_filelist();

	for(uint32_t i = 0; i < connx_operator_count; i++) {
		connx_Operator* op = &connx_operators[i];
		char prefix1[32];
		char prefix2[32];
		int prefix_len = strlen(op->name) + 1;
		sprintf(prefix1, "%s.", op->name);
		sprintf(prefix2, "%s_", op->name);
		printf("* Operator: %s\n", op->name);

		for(uint32_t j = 0; testcases[j] != NULL; j++) {
			char* testcase = testcases[j];
			if(strncmp(testcase, prefix1, prefix_len) != 0 && strncmp(testcase, prefix2, prefix_len) != 0)
				continue;

			char path[256];
			sprintf(path, "%s/%s", "testcase", testcase);
			if(!read_testcase(path)) {
				return 1;
			}

			if(!exec_testcase(op)) {
				return 1;
			}
		}
	}
	printf("* Test done\n");

	return 0;
}
