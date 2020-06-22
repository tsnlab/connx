#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <malloc.h>
#include <connx/connx.h>

#define EPSILON16 0.001
#define EPSILON32 0.00001
#define EPSILON64 0.0000001

#define EPSILON(type) (type == connx_DataType_FLOAT16 ? EPSILON16 : \
	type == connx_DataType_FLOAT32 ? EPSILON32 :					\
	type == connx_DataType_FLOAT64 ? EPSILON64 : 0)

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

bool connx_Operator_stack_update(connx_Tensor* tensor, __attribute__((unused)) int type, uint32_t idx) {
	if(stack[idx] != 0)
		connx_Tensor_delete((void*)stack[idx]);

	stack[idx] = (uintptr_t)tensor;

	return true;
}

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

char* parse_string(char *YYCURSOR, char** _str) {
	int type = 0;
	char* start = NULL;
	char* end = NULL;

	while(*YYCURSOR != 0) {
		switch(type) {
			case 0:
				switch(*YYCURSOR) {
					case '\'':
						type = 1;
						start = YYCURSOR;
						break;
					case '\"':
						type = 2;
						start = YYCURSOR;
						break;
					default:
						;
				}
				break;
			case 1:
				switch(*YYCURSOR) {
					case '\'':
						end = YYCURSOR;
						YYCURSOR++;
						goto done;
					default:
						;
				}
				break;
			case 2:
				switch(*YYCURSOR) {
					case '\"':
						end = YYCURSOR;
						YYCURSOR++;
						goto done;
					default:
						;
				}
				break;
		}

		YYCURSOR++;
	}

	return NULL;

done:

	;
	int len = (int)(end - start);
	*_str = connx_alloc(len);
	memcpy(*_str, start + 1, len - 1);
	(*_str)[len - 1] = 0;

	if(*YYCURSOR == 0)
		return NULL;
	else
		return YYCURSOR;
}

static bool exec_testcase(connx_Operator* op) {
	float epsilon = 0.0;

	char* line = readline();
	printf("\t* Test case: %s ", line);
	fflush(stdout);

	stackIdx = 1;
	line = readline();
	// Read metadata
	while(line != NULL && (line[0] == '\0' || line[0] == '@' || line[0] == '#')) {
		if(line[0] == '@') {
			if(strncmp(line, "@epsilon ", 9) == 0) {
				epsilon = strtof(line + 9, NULL);
			}
		}

		line = readline();
	}

	// Read variables
	while(line != NULL) {
		while(strchr(line, '=') == NULL)
			line = readline();

		char* id = NULL;
		char* kind = NULL;
		char* type = NULL;
		line = parse_header(line, &id, &kind, &type);

		if(strcmp(kind, "tensor") == 0) {
			connx_DataType elemType;
			if(strcmp("float16", type) == 0) {
				elemType = connx_DataType_FLOAT16;
			} else if(strcmp("float32", type) == 0) {
				elemType = connx_DataType_FLOAT32;
			} else if(strcmp("float64", type) == 0) {
				elemType = connx_DataType_FLOAT64;
			} else if(strcmp("uint8", type) == 0) {
				elemType = connx_DataType_UINT8;
			} else if(strcmp("uint16", type) == 0) {
				elemType = connx_DataType_UINT16;
			} else if(strcmp("uint32", type) == 0) {
				elemType = connx_DataType_UINT32;
			} else if(strcmp("uint64", type) == 0) {
				elemType = connx_DataType_UINT64;
			} else if(strcmp("int8", type) == 0) {
				elemType = connx_DataType_INT8;
			} else if(strcmp("int16", type) == 0) {
				elemType = connx_DataType_INT16;
			} else if(strcmp("int32", type) == 0) {
				elemType = connx_DataType_INT32;
			} else if(strcmp("int64", type) == 0) {
				elemType = connx_DataType_INT64;
			} else if(strcmp("string", type) == 0) {
				elemType = connx_DataType_STRING;
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

			if(elemType == connx_DataType_FLOAT16) {
				uint32_t i = 0;
				uint16_t* base = (uint16_t*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_float(line, &num);
						if(num != NULL) {
							base[i++] = connx_float32_to_float16(strtof(num, NULL));
						}
					}

					line = readline();
				}
			} else if(elemType == connx_DataType_FLOAT32) {
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
			} else if(elemType == connx_DataType_FLOAT64) {
				uint32_t i = 0;
				double* base = (double*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_float(line, &num);
						if(num != NULL) {
							base[i++] = strtod(num, NULL);
						}
					}

					line = readline();
				}
			} else if(elemType == connx_DataType_UINT8) {
				uint32_t i = 0;
				uint8_t* base = (uint8_t*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_int(line, &num);
						if(num != NULL) {
							base[i++] = strtoul(num, NULL, 10);
						}
					}

					line = readline();
				}
			} else if(elemType == connx_DataType_UINT16) {
				uint32_t i = 0;
				uint16_t* base = (uint16_t*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_int(line, &num);
						if(num != NULL) {
							base[i++] = strtoul(num, NULL, 10);
						}
					}

					line = readline();
				}
			} else if(elemType == connx_DataType_UINT32) {
				uint32_t i = 0;
				uint32_t* base = (uint32_t*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_int(line, &num);
						if(num != NULL) {
							base[i++] = strtoul(num, NULL, 10);
						}
					}

					line = readline();
				}
			} else if(elemType == connx_DataType_UINT64) {
				uint32_t i = 0;
				uint64_t* base = (uint64_t*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_int(line, &num);
						if(num != NULL) {
							base[i++] = strtoull(num, NULL, 10);
						}
					}

					line = readline();
				}
			} else if(elemType == connx_DataType_INT8) {
				uint32_t i = 0;
				int8_t* base = (int8_t*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_int(line, &num);
						if(num != NULL) {
							base[i++] = strtol(num, NULL, 10);
						}
					}

					line = readline();
				}
			} else if(elemType == connx_DataType_INT16) {
				uint32_t i = 0;
				int16_t* base = (int16_t*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_int(line, &num);
						if(num != NULL) {
							base[i++] = strtol(num, NULL, 10);
						}
					}

					line = readline();
				}
			} else if(elemType == connx_DataType_INT32) {
				uint32_t i = 0;
				int32_t* base = (int32_t*)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* num = NULL;
						line = parse_int(line, &num);
						if(num != NULL) {
							base[i++] = strtol(num, NULL, 10);
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
			} else if(elemType == connx_DataType_STRING) {
				uint32_t i = 0;
				char** base = (char**)tensor->base;

				line = readline();
				while(line != NULL && strchr(line, '=') == NULL) {
					while(line != NULL) {
						char* str = NULL;
						line = parse_string(line, &str);
						if(str != NULL) {
							base[i++] = str;
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
		} else if(strcmp(kind, "vararg") == 0 || strcmp(kind, "varar") == 0) {
			char* num = NULL;
			parse_int(line, &num);

			stack[stackIdx++] = (uintptr_t)strtol(num, NULL, 10);

			line = readline();
		} else if(strcmp(kind, "nul") == 0 || strcmp(kind, "null") == 0) {
			stack[stackIdx++] = 0;
		} else {
			fprintf(stderr, "Illegal variable kind: %s\n", kind);
			abort();
		}
	}

	stack[0] = stackIdx - 1;

	/*
	if(op->isInputVarArgs || op->isOutputVarArgs) {
		if(stack[0] < 1 + op->outputCount + op->inputCount + op->attributeCount) {
			printf(RED "\tSTACK COUNT TOO SMALL: expected: more than %u but %lu\n" END, op->outputCount + op->inputCount + op->attributeCount, stack[0]);
			return false;
		}
	} else {
		if(stack[0] != op->outputCount + op->inputCount + op->attributeCount) {
			printf(RED "\tSTACK COUNT MISMATCH: expected: %u but %lu\n" END, op->outputCount + op->inputCount + op->attributeCount, stack[0]);
			return false;
		}
	}
	*/

	if(!op->resolve(stack)) {
		printf(RED "\tRESOLVE FAILED: %s\n" END, connx_exception_message());
		return false;
	}

	if(!op->exec(stack)) {
		printf(RED "\tEXECUTION FAILED: %s\n" END, connx_exception_message());
		return false;
	}

	connx_Tensor* output = (connx_Tensor*)stack[1];
	connx_Tensor* result= (connx_Tensor*)stack[stackIdx - 1];
	if(epsilon == 0.0)
		epsilon = EPSILON(output->elemType);

	if(connx_Tensor_isNearlyEquals(output, result, epsilon)) {
		printf(GREEN "\tPASS\n" END);
	} else {
		printf(RED "\tFAILED\n" END);
		connx_Tensor_dump_compare(output, result, epsilon);
	}

	// clean up
	uint32_t stackIdx2 = 1;

	// delete outputs
	uintptr_t output_count = op->outputCount;

	if(op->isOutputVarArgs) {
		output_count = stack[stackIdx2++];
	}

	for(uint32_t i = 0; i < output_count; i++, stackIdx2++) {
		if(stack[stackIdx2] != 0) {
			connx_Tensor* tensor = (connx_Tensor*)stack[stackIdx2];
			connx_Tensor_delete(tensor);
		}
	}

	// delete inputs
	uintptr_t input_count = op->inputCount;

	if(op->isInputVarArgs) {
		input_count = stack[stackIdx2++];
	}

	for(uint32_t i = 0; i < input_count; i++, stackIdx2++) {
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
