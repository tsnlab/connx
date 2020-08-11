#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <connx/connx.h>
#include <connx/backend.h>

#if DEBUG
#include <connx/dump.h>
#endif /* DEBUG */

// internal functions
bool opset_delete(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	uint32_t input_count = CONNX_INPUT_COUNT(counts);

	for(uint32_t i = 0; i < input_count; i++) {
		if(params[i] >= backend->initializer_count) {
#if DEBUG
			printf("DEBUG: delete: %u\n", CONNX_GET_INPUT_INDEX(i));
#endif /* DEBUG */
			connx_Backend_delete_variable(backend, CONNX_GET_INPUT_INDEX(i));
		}
	}

	return true;
}

bool opset_input(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	// Do nothing
	return true;
}

bool opset_output(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	// Do nothing
	return true;
}

// DataType
uint32_t connx_DataType_size(connx_DataType type) {
	static const uint32_t DATA_TYPE_SIZE[] = { 0, 4, 1, 1, 2, 2, 4, 8, 0, 1, 2, 8, 4, 4 };

	if(type < 0 || type >= sizeof(DATA_TYPE_SIZE))
		return 0;
	else
		return DATA_TYPE_SIZE[type];
}

const char* connx_DataType_name(connx_DataType type) {
	static const char* DATA_TYPE_NAME[] = {
		"void", "float32", "uint8", "int8",
		"uint16", "int16", "int32", "int64",
		"string", "bool", "float16", "float64",
		"uint32", "uint64",
	};

	if(type < 0 || type >= sizeof(DATA_TYPE_NAME))
		return "unknown";
	else
		return DATA_TYPE_NAME[type];
}

// Tensor

connx_Tensor* connx_Tensor_create(connx_HAL* hal, connx_DataType type, uint32_t dimension, uint32_t* lengths) {
	uint32_t total = 1;
	for(uint32_t i = 0; i < dimension; i++) {
		total *= lengths[i];
	}

	connx_Tensor* tensor = hal->alloc(hal, sizeof(connx_Tensor) + connx_DataType_size(type) * total);
	if(tensor == NULL) {
		return NULL;
	}

	tensor->type = type;

	tensor->dimension = dimension;

	tensor->lengths = hal->alloc(hal, sizeof(uint32_t) * dimension);
	if(tensor->lengths == NULL) {
		hal->free(hal, tensor);
		return NULL;
	}

	memcpy(tensor->lengths, lengths, sizeof(uint32_t) * dimension);

	return tensor;
}

connx_Tensor* connx_Tensor_create_from_buffer(connx_HAL* hal, void* buf) {
	uint32_t data_type = *(uint32_t*)buf;
	if(data_type == 0)	// null tensor
		return NULL;

	buf += sizeof(uint32_t);

	uint32_t dimension = *(uint32_t*)buf;
	buf += sizeof(uint32_t);

	uint32_t total = 1;
	uint32_t lengths[dimension];
	for(uint32_t j = 0; j < dimension; j++) {
		lengths[j] = *(uint32_t*)buf;
		buf += sizeof(uint32_t);

		total *= lengths[j];
	}

	connx_Tensor* tensor = connx_Tensor_create(hal, data_type, dimension, lengths);
	memcpy(tensor->base, buf, connx_DataType_size(data_type) * total);

	return tensor;
}

void connx_Tensor_delete(connx_HAL* hal, connx_Tensor* tensor) {
	hal->free(hal, tensor->lengths);
	hal->free(hal, tensor);
}

bool connx_Tensor_is_shape_equals(connx_Tensor* x, connx_Tensor* y) {
	if(x->type != y->type)
		return false;

	if(x->dimension != y->dimension)
		return false;

	for(uint32_t i = 0; i < x->dimension; i++) {
		if(x->lengths[i] != y->lengths[i])
			return false;
	}

	return true;
}

uint32_t connx_Tensor_total(connx_Tensor* tensor) {
	uint32_t total = 1;
	for(uint32_t i = 0; i < tensor->dimension; i++) {
		total *= tensor->lengths[i];
	}

	return total;
}

int32_t connx_Tensor_accuracy(connx_Tensor* x, connx_Tensor* y) {
	if(!connx_Tensor_is_shape_equals(x, y))
		return -2;

	uint32_t count = connx_Tensor_total(x);

	switch(x->type) {
		case connx_UINT8:
		case connx_UINT16:
		case connx_UINT32:
		case connx_UINT64:
		case connx_INT8:
		case connx_INT16:
		case connx_INT32:
		case connx_INT64:
		case connx_BOOL:
		case connx_FLOAT16:
			return memcmp(x->base, y->base, connx_DataType_size(x->type) * count) == 0;
			/* float16
			{
				uint16_t* base = (uint16_t*)x->base;
				uint16_t* base2 = (uint16_t*)y->base;
				float e = epsilon;

				for(uint32_t i = 0; i < count; i++) {
					if(base[i] == base2[i])
						continue;

					float diff = connx_float16_to_float32(base[i]) - connx_float16_to_float32(base2[i]);
					if(diff >= -e && diff <= e)
						continue;

					return false;
				}

				return true;
			}
			*/
		case connx_FLOAT32:
			{
				float* base = (float*)x->base;
				float* base2 = (float*)y->base;

				for(int32_t precision = 6; precision > 0; precision--) {
					float epsilon = powf(10, -precision);

					for(uint32_t i = 0; i < count; i++) {
						if(base[i] == base2[i])
							continue;

						if(isnan(base[i]) && isnan(base2[i]))
							continue;

						float diff = base[i] - base2[i];
						if(diff >= -epsilon && diff <= epsilon)
							continue;

						goto next_float32;
					}

					return precision;
next_float32:
					;
				}

				return -1;
			}
		case connx_FLOAT64:
			{
				double* base = (double*)x->base;
				double* base2 = (double*)y->base;

				for(int32_t precision = 15; precision > 0; precision--) {
					double epsilon = 10^-precision;

					for(uint32_t i = 0; i < count; i++) {
						if(base[i] == base2[i])
							continue;

						if(isnan(base[i]) && isnan(base2[i]))
							continue;

						double diff = base[i] - base2[i];
						if(diff >= -epsilon && diff <= epsilon)
							continue;

						goto next_float64;
					}

					return precision;
next_float64:
					;
				}

				return -1;
			}
		case connx_STRING:
			{
				char** base = (char**)x->base;
				char** base2 = (char**)y->base;

				for(uint32_t i = 0; i < count; i++) {
					if(strcmp(base[i], base2[i]) != 0) {
						return 0;
					}
				}
				return -1;
			}
		default:
			;
			return -1;
	}
}

// Backend
connx_Call* connx_Call_create(connx_HAL* hal, connx_Operator op, uint32_t counts) {
	connx_Call* call = hal->alloc(hal, sizeof(connx_Call));
	call->op = op;
	call->counts = counts;
	call->params = hal->alloc(hal, sizeof(uint32_t) * CONNX_TOTAL_COUNT(counts));

	return call;
}

void connx_Call_delete(connx_HAL* hal, connx_Call* call) {
	if(call->params != NULL) {
		hal->free(hal, call->params);
	}

	hal->free(hal, call);
}

connx_Path* connx_Path_create(connx_HAL* hal) {
	connx_Path* path = hal->alloc(hal, sizeof(connx_Path));

	return path;
}

void connx_Path_delete(connx_HAL* hal, connx_Path* path) {
	if(path->calls != NULL) {
		for(uint32_t i = 0; i < path->call_count; i++) {
			if(path->calls[i] != NULL) {
				connx_Call_delete(hal, path->calls[i]);
			}
		}

		hal->free(hal, path->calls);
	}

	if(path->output_paths != NULL) {
		hal->free(hal, path->output_paths);
	}

	if(path->input_paths != NULL) {
		hal->free(hal, path->input_paths);
	}

	hal->free(hal, path);
}

bool connx_Path_run(connx_Path* path, connx_Backend* backend) {
	for(uint32_t i = 0; i < path->call_count; i++) {
		connx_Call* call = path->calls[i];

#if DEBUG
		connx_Call_dump(backend->hal, call);
		backend->hal->debug_tab++;
#endif /* DEBUG */

		if(!call->op(backend, call->counts, call->params)) {
			return false;
		}

#if DEBUG
		uint32_t output_count = CONNX_OUTPUT_COUNT(call->counts);
		for(uint32_t i = 0; i < output_count; i++) {
			backend->hal->debug(backend->hal, "output[%u] = ", call->params[i]);
			if(backend->variables[call->params[i]] == NULL)
				backend->hal->debug(backend->hal, "null\n");
			else
				connx_Tensor_dump(backend->hal, backend->variables[call->params[i]]);
		}
		backend->hal->debug_tab--;
#endif /* DEBUG */
	}

	return true;
}

#define TOKEN_COUNT 32

static uint32_t parse_line(char** script, char** tokens) {
restart:

	if(*script == NULL) {
		return 0;
	}

	uint32_t count = 0;

	// start - start of line
	char* start = *script;
	while(*start == ' ' || *start == '\t')
		start++;

	// end - end of line
	char* end = start;
	while(*end != '\n' && *end != '\0')
		end++;

	// next script
	if(*end == '\n') {
		*end = '\0';
		*script = end + 1;
	} else {
		*script = NULL;
	}

	// jump one line comment or empty string
	if(*start == '#' || *start == '\0')
		goto restart;

	// remove comment
	char* pos = start;
	while(pos < end && *pos != '#')
		pos++;

	if(*pos == '#') {
		*pos = '\0';
		end = pos;
	}

	// remove right spaces
	pos = end - 1;
	while(pos >= start && *pos == ' ')
		*pos-- = '\0';

	// parse tokens
	while(count < TOKEN_COUNT) {
		// pos - end of token
		pos = start;
		while(*pos != ' ' && *pos != '\0')
			pos++;

		if(*pos == ' ') {
			*pos = '\0';
			tokens[count++] = start;
			start = pos + 1;

			while(*start == ' ')
				start++;
		} else {
			tokens[count++] = start;
			break;
		}
	}

	return count;
}

static bool parse_script(connx_Backend* backend) {
	connx_HAL* hal = backend->hal;
	char* script = (char*)hal->load(hal, "main.cnx");
	if(script == NULL) {
		return false;
	}

	char* orig_script = script;

	/**
	 * status
	 * 0 - model
	 * 1 - path
	 */
	int status = 0;
	char* tokens[TOKEN_COUNT];
	connx_Path* path = NULL;
	uint32_t path_call_idx = 0;

	uint32_t count = parse_line(&script, tokens);
	while(count > 0) {
		if(status == 0) {
			if(strncmp(tokens[0], "opset", 5) == 0) {
				if(count < 2) {
					hal->error(hal, "Unexpected parameter number for opset: %u\n", 2);
					hal->unload(hal, orig_script);
					return false;
				}

				backend->opset = strtoul(tokens[1], NULL, 0);
			} else if(strncmp(tokens[0], "initializers", 13) == 0) {
				if(count < 2) {
					hal->error(hal, "Unexpected parameter number for initializers: %u\n", 2);
					hal->unload(hal, orig_script);
					return false;
				}

				backend->initializer_count = strtoul(tokens[1], NULL, 0);
			} else if(strncmp(tokens[0], "variables", 10) == 0) {
				if(count < 2) {
					hal->error(hal, "Unexpected parameter number for variables: %u\n", 2);
					hal->unload(hal, orig_script);
					return false;
				}

				uint32_t variable_count = strtoul(tokens[1], NULL, 0);
				backend->variable_count = backend->initializer_count + variable_count;
				backend->variables = hal->alloc(hal, sizeof(connx_Tensor*) * backend->variable_count);

				if(backend->variables == NULL) {
					hal->error(hal, "Out of memory\n");
					hal->unload(hal, orig_script);
					return false;
				}
			} else if(strncmp(tokens[0], "paths", 6) == 0) {
				if(count < 2) {
					hal->error(hal, "Unexpected parameter number for paths: %u\n", 2);
					hal->unload(hal, orig_script);
					return false;
				}

				backend->path_count = strtoul(tokens[1], NULL, 0);
				backend->paths = hal->alloc(hal, sizeof(connx_Path) * backend->path_count);

				if(backend->paths == NULL) {
					hal->error(hal, "Out of memory\n");
					hal->unload(hal, orig_script);
					return false;
				}
			} else if(strncmp(tokens[0], "path", 5) == 0) {
				if(count < 2) {
					hal->error(hal, "Unexpected parameter number for initializers: %u\n", 2);
					hal->unload(hal, orig_script);
					return false;
				}

				uint32_t id = strtoul(tokens[1], NULL, 0);

				path = connx_Path_create(hal);
				path_call_idx = 0;
				backend->paths[id] = path;

				status = 1;
			} else if(strncmp(tokens[0], "start", 6) == 0) {
				backend->start_count = count - 1;

				backend->starts = hal->alloc(hal, sizeof(uint32_t) * backend->start_count);
				for(uint32_t i = 0; i < backend->start_count; i++) {
					backend->starts[i] = strtoul(tokens[1 + i], NULL, 0);
				}
			} else if(strncmp(tokens[0], "stop", 6) == 0) {
				backend->stop_count = count - 1;

				backend->stops = hal->alloc(hal, sizeof(uint32_t) * backend->stop_count);
				for(uint32_t i = 0; i < backend->stop_count; i++) {
					backend->stops[i] = strtoul(tokens[1 + i], NULL, 0);
				}

				break;	// end of file
			} else {
				hal->error(hal, "Not supported command: '%s'\n", tokens[0]);
				hal->unload(hal, orig_script);
				return false;
			}
		} else if(status == 1) {
			if(strncmp(tokens[0], "input_paths", 12) == 0) {
				path->input_path_count = count - 1;
				path->input_paths = hal->alloc(hal, sizeof(uint32_t) * path->input_path_count);
				for(uint32_t i = 0; i < path->input_path_count; i++) {
					path->input_paths[i] = strtoul(tokens[1 + i], NULL, 0);
				}
			} else if(strncmp(tokens[0], "output_paths", 13) == 0) {
				path->output_path_count = count - 1;
				path->output_paths = hal->alloc(hal, sizeof(uint32_t) * path->output_path_count);
				for(uint32_t i = 0; i < path->output_path_count; i++) {
					path->output_paths[i] = strtoul(tokens[1 + i], NULL, 0);
				}
			} else if(strncmp(tokens[0], "calls", 6) == 0) {
				if(count < 2) {
					hal->error(hal, "Unexpected parameter number for calls: %u\n", 2);
					hal->unload(hal, orig_script);
					return false;
				}

				path->call_count = strtoul(tokens[1], NULL, 0);
				path->calls = hal->alloc(hal, sizeof(connx_Call) * path->call_count);
			} else if(strncmp(tokens[0], "call", 5) == 0) {
				if(count < 5) {
					hal->error(hal, "Unexpected parameter number for call: %u\n", 5);
					hal->unload(hal, orig_script);
					return false;
				}

				char* op_name = tokens[1];
				connx_Operator op = NULL;
				if(strcmp(op_name, "delete") == 0) {
					op = opset_delete;
				} else if(strcmp(op_name, "input") == 0) {
					op = opset_input;
				} else if(strcmp(op_name, "output") == 0) {
					op = opset_output;
				} else {
					for(uint32_t i = 0; connx_operator_names[i] != NULL; i++) {
						if(strcmp(op_name, connx_operator_names[i]) == 0) {
							op = connx_operators[i];
							break;
						}
					}
				}

				if(op == NULL) {
					hal->error(hal, "Cannot find operator %s\n", op_name);
					hal->unload(hal, orig_script);
					return false;
				}

				uint32_t output_count = strtoul(tokens[2], NULL, 0);
				uint32_t input_count = strtoul(tokens[3], NULL, 0);
				uint32_t attribute_count = strtoul(tokens[4], NULL, 0);

				connx_Call* call = connx_Call_create(hal, op, CONNX_COUNTS(output_count, input_count, attribute_count));
				uint32_t total = output_count + input_count + attribute_count;
				for(uint32_t i = 0; i < total; i++) {
					call->params[i] = strtoul(tokens[5 + i], NULL, 0);
				}

				path->calls[path_call_idx++] = call;

				if(path_call_idx >= path->call_count) {
					status = 0;
				}
			} else if(strncmp(tokens[0], "delete", 7) == 0) {
				connx_Call* call = connx_Call_create(hal, opset_delete, CONNX_COUNTS(0, count - 1, 0));
				for(uint32_t i = 0; i < count - 1; i++) {
					call->params[i] = strtoul(tokens[1 + i], NULL, 0);
				}

				path->calls[path_call_idx++] = call;
			} else {
				hal->error(hal, "Not supported command: %s\n", tokens[0]);
				hal->unload(hal, orig_script);
				return false;
			}
		} else {
			hal->error(hal, "Illegal script parser status: %d\n", status);
			hal->unload(hal, orig_script);
			return false;
		}

		count = parse_line(&script, tokens);
	}

	hal->unload(hal, orig_script);
	return true;
}

static bool load_initializers(connx_Backend* backend) {
	connx_HAL* hal = backend->hal;

	uint32_t* index = hal->load(hal, "init.idx");
	if(index == NULL) {
		backend->hal->error(hal, "Cannot load init.idx\n");
		return false;
	}

	void* db = hal->load(hal, "init.db");
	if(db == NULL) {
		hal->unload(hal, index);
		backend->hal->error(hal, "Cannot load init.db\n");
		return false;
	}

	for(uint32_t i = 0; i < backend->initializer_count; i++) {
		void* info = db + index[i];

		backend->variables[i] = connx_Tensor_create_from_buffer(hal, info);
	}

	hal->unload(hal, index);
	hal->unload(hal, db);

	return true;
}

static bool load_attributes(connx_Backend* backend) {
	connx_HAL* hal = backend->hal;

	uint32_t* index = hal->load(hal, "attr.idx");
	if(index == NULL) {
		backend->hal->error(hal, "Cannot load attr.idx\n");
		return false;
	}

	void* db = hal->load(hal, "attr.db");
	if(index == NULL) {
		hal->unload(hal, index);
		backend->hal->error(hal, "Cannot load attr.db\n");
		return false;
	}

	backend->attribute_index = index;
	backend->attributes = db;

	return true;
}

connx_Backend* connx_Backend_create(connx_HAL* hal) {
	connx_Backend* backend = hal->alloc(hal, sizeof(connx_Backend));
	if(backend == NULL) {
		return NULL;
	}

	backend->hal = hal;
	if(!parse_script(backend)) {
		connx_Backend_delete(backend);
		return NULL;
	}

	if(!load_initializers(backend)) {
		connx_Backend_delete(backend);
		return NULL;
	}

	if(!load_attributes(backend)) {
		connx_Backend_delete(backend);
		return NULL;
	}

	return backend;
}

void connx_Backend_delete(connx_Backend* backend) {
	connx_HAL* hal = backend->hal;

	if(backend->stops != NULL) {
		hal->free(hal, backend->stops);
	}

	if(backend->starts != NULL) {
		hal->free(hal, backend->starts);
	}

	if(backend->attributes != NULL) {
		hal->unload(hal, backend->attributes);
	}

	if(backend->attribute_index != NULL) {
		hal->unload(hal, backend->attribute_index);
	}

	if(backend->variables != NULL) {
		// ignore inputs and outputs: user has responsible to manage inputs and outputs
		if(backend->paths != NULL) {
			connx_Path* input_path = backend->paths[0];
			connx_Call* input_call = input_path->calls[0];
			uint32_t index_count = CONNX_OUTPUT_COUNT(input_call->counts);
			uint32_t* index = input_call->params;

			for(uint32_t i = 0; i < index_count; i++) {
				backend->variables[index[i]] = NULL;
			}

			connx_Path* output_path = backend->paths[backend->path_count - 1];
			connx_Call* output_call = output_path->calls[output_path->call_count - 1];
			index_count = CONNX_INPUT_COUNT(output_call->counts);
			index = output_call->params + CONNX_OUTPUT_COUNT(output_call->counts);

			for(uint32_t i = 0; i < index_count; i++) {
				backend->variables[index[i]] = NULL;
			}
		}

		for(uint32_t i = 0; i < backend->variable_count; i++) {
			if(backend->variables[i] != NULL) {
				connx_Tensor_delete(hal, backend->variables[i]);
			}
		}

		hal->free(hal, backend->variables);
	}

	if(backend->paths != NULL) {
		for(uint32_t i = 0; i < backend->path_count; i++) {
			if(backend->paths[i] != NULL) {
				connx_Path_delete(hal, backend->paths[i]);
			}
		}

		hal->free(hal, backend->paths);
	}

	hal->free(hal, backend);
}

bool connx_Backend_run(connx_Backend* backend, uint32_t* output_count, connx_Tensor** outputs, uint32_t input_count, connx_Tensor** inputs) {
	// input
	connx_Path* input_path = backend->paths[0];
	connx_Call* input_call = input_path->calls[0];
	uint32_t index_count = CONNX_OUTPUT_COUNT(input_call->counts);
	uint32_t* index = input_call->params;

	if(input_count != index_count) {
		backend->hal->error(backend->hal, "Input parameter mismatch: %u, expected: %u\n", input_count, index_count);
		return false;
	}

	for(uint32_t i = 0; i < index_count && inputs[i] != NULL; i++) {
		connx_Backend_set_variable(backend, index[i], inputs[i]);
	}

	// run
	for(uint32_t i = 0; i < backend->path_count; i++) {
		connx_Path* path = backend->paths[i];

		if(!connx_Path_run(path, backend)) {
			return false;
		}
	}

	// forget inputs: User is responsible to manage inputs
	for(uint32_t i = 0; i <  index_count; i++) {
		backend->variables[index[i]] = NULL;
	}

	// output
	connx_Path* output_path = backend->paths[backend->path_count - 1];
	connx_Call* output_call = output_path->calls[output_path->call_count - 1];
	index_count = CONNX_INPUT_COUNT(output_call->counts);
	index = output_call->params + CONNX_OUTPUT_COUNT(output_call->counts);

	if(index_count < *output_count)
		*output_count = index_count;

	for(uint32_t i = 0; i < index_count; i++) {
		if(i < *output_count) {
			outputs[i] = backend->variables[index[i]];
			backend->variables[index[i]] = NULL;	// forget outputs: user is responsible to manage outputs
		} else {
			connx_Backend_delete_variable(backend, index[i]);
		}
	}

	return true;
}

connx_Tensor* connx_Backend_load_tensor(connx_Backend* backend, const char* name) {
	connx_HAL* hal = backend->hal;

	void* data = hal->load(hal, name);
	if(data == NULL) {
		backend->hal->error(hal, "Cannot load tensor: '%s'\n", name);
		return NULL;
	}

	connx_Tensor* tensor = connx_Tensor_create_from_buffer(hal, data);

	hal->unload(hal, data);

	return tensor;
}

bool connx_Backend_has_variable(connx_Backend* backend, uint32_t id) {
	return id < backend->variable_count && backend->variables[id] != NULL;
}

connx_Tensor* connx_Backend_get_variable(connx_Backend* backend, uint32_t id) {
	if(id < backend->variable_count)
		return backend->variables[id];
	else
		return NULL;
}

bool connx_Backend_set_variable(connx_Backend* backend, uint32_t id, connx_Tensor* tensor) {
	if(id >= backend->variable_count)
		return false;

	if(backend->variables[id] != NULL) {
		connx_Tensor_delete(backend->hal, backend->variables[id]);
	}

	backend->variables[id] = tensor;

	return true;
}

bool connx_Backend_delete_variable(connx_Backend* backend, uint32_t id) {
	if(id >= backend->variable_count)
		return false;

	if(backend->variables[id] != NULL) {
		connx_Tensor_delete(backend->hal, backend->variables[id]);
		backend->variables[id] = NULL;
		return true;
	} else {
		return true;
	}
}

void* connx_Backend_get_attribute(connx_Backend* backend, uint32_t id) {
	uint32_t offset = backend->attribute_index[id];
	return backend->attributes + offset;
}
