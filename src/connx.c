#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <connx/connx.h>
#include <connx/backend.h>

#define MESSAGE_SIZE 256
static char _message[MESSAGE_SIZE];

// Delete operator
static bool opset_delete(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	uint32_t input_count = CONNX_INPUT_COUNT(counts);

	for(uint32_t i = 0; i < input_count; i++) {
		connx_Backend_delete_variable(backend, params[i]);
	}

	return true;
}

// DataType
uint32_t connx_DataType_size(connx_DataType type) {
	const static uint32_t DATA_TYPE_SIZE[] = { 0, 4, 1, 1, 2, 2, 4, 8, 0, 1, 2, 8, 4, 4 };

	return DATA_TYPE_SIZE[type];
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

	if(path->outputs != NULL) {
		hal->free(hal, path->outputs);
	}

	if(path->inputs != NULL) {
		hal->free(hal, path->inputs);
	}

	hal->free(hal, path);
}

bool connx_Path_run(connx_Path* path, connx_Backend* backend) {
	for(uint32_t i = 0; i < path->call_count; i++) {
		connx_Call* call = path->calls[i];
		if(!call->op(backend, call->counts, call->params)) {
			return false;
		}
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
					snprintf(_message, MESSAGE_SIZE, "Unexpected parameter number for opset: %u", 2);
					hal->error(hal, _message);
					hal->unload(hal, orig_script);
					return false;
				}

				backend->opset = strtoul(tokens[1], NULL, 0);
			} else if(strncmp(tokens[0], "paths", 6) == 0) {
				if(count < 2) {
					snprintf(_message, MESSAGE_SIZE, "Unexpected parameter number for paths: %u", 2);
					hal->error(hal, _message);
					hal->unload(hal, orig_script);
					return false;
				}

				backend->path_count = strtoul(tokens[1], NULL, 0);
				backend->paths = hal->alloc(hal, sizeof(connx_Path) * backend->path_count);

				if(backend->paths == NULL) {
					snprintf(_message, MESSAGE_SIZE, "Out of memory");
					hal->error(hal, _message);
					hal->unload(hal, orig_script);
					return false;
				}
			} else if(strncmp(tokens[0], "initializers", 13) == 0) {
				if(count < 2) {
					snprintf(_message, MESSAGE_SIZE, "Unexpected parameter number for initializers: %u", 2);
					hal->error(hal, _message);
					hal->unload(hal, orig_script);
					return false;
				}

				backend->initializer_count = strtoul(tokens[1], NULL, 0);
			} else if(strncmp(tokens[0], "variables", 10) == 0) {
				if(count < 2) {
					snprintf(_message, MESSAGE_SIZE, "Unexpected parameter number for variables: %u", 2);
					hal->error(hal, _message);
					hal->unload(hal, orig_script);
					return false;
				}

				uint32_t variable_count = strtoul(tokens[1], NULL, 0);
				backend->variable_count = backend->initializer_count + variable_count;
				backend->variables = hal->alloc(hal, sizeof(connx_Tensor*) * backend->variable_count);

				if(backend->variables == NULL) {
					snprintf(_message, MESSAGE_SIZE, "Out of memory");
					hal->error(hal, _message);
					hal->unload(hal, orig_script);
					return false;
				}
			} else if(strncmp(tokens[0], "path", 5) == 0) {
				if(count < 2) {
					snprintf(_message, MESSAGE_SIZE, "Unexpected parameter number for initializers: %u", 2);
					hal->error(hal, _message);
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
			} else if(strncmp(tokens[0], "clean", 6) == 0) {
				backend->clean_count = count - 1;

				backend->cleans = hal->alloc(hal, sizeof(uint32_t) * backend->clean_count);
				for(uint32_t i = 0; i < backend->clean_count; i++) {
					backend->cleans[i] = strtoul(tokens[1 + i], NULL, 0);
				}
			} else {
				snprintf(_message, MESSAGE_SIZE, "Not supported command: '%s'", tokens[0]);
				hal->error(hal, _message);
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
					snprintf(_message, MESSAGE_SIZE, "Unexpected parameter number for calls: %u", 2);
					hal->error(hal, _message);
					hal->unload(hal, orig_script);
					return false;
				}

				path->call_count = strtoul(tokens[1], NULL, 0) * 2;	// call + delete => call_count * 2
				path->calls = hal->alloc(hal, sizeof(connx_Call) * path->call_count);
			} else if(strncmp(tokens[0], "call", 5) == 0) {
				if(count < 5) {
					snprintf(_message, MESSAGE_SIZE, "Unexpected parameter number for call: %u", 5);
					hal->error(hal, _message);
					hal->unload(hal, orig_script);
					return false;
				}

				char* op_name = tokens[1];
				connx_Operator op = NULL;
				for(uint32_t i = 0; connx_operator_names[i] != NULL; i++) {
					if(strcmp(op_name, connx_operator_names[i]) == 0) {
						op = connx_operators[i];
						break;
					}
				}

				if(op == NULL) {
					snprintf(_message, MESSAGE_SIZE, "Cannot find operator %s", op_name);
					hal->error(hal, _message);
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
			} else if(strncmp(tokens[0], "delete", 7) == 0) {
				connx_Call* call = connx_Call_create(hal, opset_delete, CONNX_COUNTS(0, count - 1, 0));
				for(uint32_t i = 0; i < count - 1; i++) {
					call->params[i] = strtoul(tokens[1 + i], NULL, 0);
				}

				path->calls[path_call_idx++] = call;
			} else if(strncmp(tokens[0], "inputs", 7) == 0) {
				path->input_count = count - 1;

				path->inputs = hal->alloc(hal, sizeof(uint32_t) * path->input_count);
				for(uint32_t i = 0; i < count - 1; i++) {
					path->inputs[i] = strtoul(tokens[1 + i], NULL, 0);
				}
			} else if(strncmp(tokens[0], "outputs", 8) == 0) {
				path->output_count = count - 1;

				path->outputs = hal->alloc(hal, sizeof(uint32_t) * path->output_count);
				for(uint32_t i = 0; i < count - 1; i++) {
					path->outputs[i] = strtoul(tokens[1 + i], NULL, 0);
				}

				status = 0;
			} else {
				snprintf(_message, MESSAGE_SIZE, "Not supported command: %s", tokens[0]);
				hal->error(hal, _message);
				hal->unload(hal, orig_script);
				return false;
			}
		} else {
			snprintf(_message, MESSAGE_SIZE, "Illegal script parser status: %d", status);
			hal->error(hal, _message);
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
		snprintf(_message, MESSAGE_SIZE, "Cannot load init.idx");
		backend->hal->error(hal, _message);
		return false;
	}

	void* db = hal->load(hal, "init.db");
	if(db == NULL) {
		hal->unload(hal, index);
		snprintf(_message, MESSAGE_SIZE, "Cannot load init.db");
		backend->hal->error(hal, _message);
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
		snprintf(_message, MESSAGE_SIZE, "Cannot load attr.idx");
		backend->hal->error(hal, _message);
		return false;
	}

	void* db = hal->load(hal, "attr.db");
	if(index == NULL) {
		hal->unload(hal, index);
		snprintf(_message, MESSAGE_SIZE, "Cannot load attr.db");
		backend->hal->error(hal, _message);
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

	if(backend->cleans != NULL) {
		hal->free(hal, backend->cleans);
	}

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

connx_Tensor** connx_Backend_run(connx_Backend* backend, connx_Tensor** inputs) {
	// Check output count exceeds
	if(backend->clean_count >= CONNX_MAX_BACKEND_OUTPUT_COUNT - 1) {
		snprintf(_message, MESSAGE_SIZE, "Output tensor count exceeds: %u > %u", backend->clean_count, CONNX_MAX_BACKEND_OUTPUT_COUNT - 1);
		return NULL;
	}

	// input
	uint32_t index_count = backend->paths[0]->output_count;
	uint32_t* index = backend->paths[0]->outputs;

	for(uint32_t i = 0; i < index_count && inputs[i] != NULL; i++) {
		connx_Backend_set_variable(backend, index[i], inputs[i]);
	}

	// run
	for(uint32_t i = 0; i < backend->path_count; i++) {
		connx_Path* path = backend->paths[i];

		if(!connx_Path_run(path, backend)) {
			return NULL;
		}
	}

	// clean variables
	for(uint32_t i = backend->initializer_count; i < backend->variable_count; i++) {
		connx_Backend_delete_variable(backend, i);
	}

	// set outputs
	for(uint32_t i = 0; i < backend->clean_count && i < CONNX_MAX_BACKEND_OUTPUT_COUNT; i++) {
		backend->outputs[i] = backend->variables[backend->cleans[i]];
	}

	backend->outputs[backend->clean_count] = NULL;
	printf("output ptr: %p\n", backend->outputs);

	return backend->outputs;
}

void connx_Backend_clean(connx_Backend* backend) {
	for(uint32_t i = 0; i < backend->clean_count; i++) {
		connx_Backend_delete_variable(backend, backend->cleans[i]);
	}
}

connx_Tensor* connx_Backend_load_tensor(connx_Backend* backend, const char* name) {
	connx_HAL* hal = backend->hal;

	void* data = hal->load(hal, name);
	if(data == NULL) {
		snprintf(_message, MESSAGE_SIZE, "Cannot load tensor: '%s'", name);
		backend->hal->error(hal, _message);
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
		return true;
	} else {
		return true;
	}
}

void* connx_Backend_get_attribute(connx_Backend* backend, uint32_t id) {
	uint32_t offset = backend->attribute_index[id];
	return backend->attributes + offset;
}
