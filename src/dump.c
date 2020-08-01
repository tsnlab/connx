#include <stdio.h>
#include <connx/dump.h>

#define MESSAGE_SIZE 256
static char _message[MESSAGE_SIZE];

static char* find_op_name(connx_Operator op) {
	for(int i = 0; connx_operators[i] != NULL; i++) {
		if(connx_operators[i] == op) {
			return connx_operator_names[i];
		}
	}

	return "NOP";
}

void connx_Call_dump(connx_HAL* hal, connx_Call* call, int depth) {
	hal->info(hal, "call");
}

void connx_Path_dump(connx_HAL* hal, connx_Path* path, int depth) {
	hal->info(hal, "path");
}

void connx_Backend_dump(connx_HAL* hal, connx_Backend* backend, int depth) {
	hal->info(hal, "backend");
}

