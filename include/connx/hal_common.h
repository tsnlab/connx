/*
 *  CONNX, C implementation of Open Neural Network Exchange Runtime
 *  Copyright (C) 2019-2022 TSN Lab, Inc.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef __CONNX_HAL_COMMON_H__
#define __CONNX_HAL_COMMON_H__

#include <stdbool.h>
#include <stdint.h>

// Ref: https://iq.opengenus.org/detect-operating-system-in-c/
#ifdef __linux__
#include <pthread.h>
#endif /* __linux__ */

#define CONNX_ALIGNMENT 16 // Data alignment
#define CONNX_ALIGN(offset) (((offset) + CONNX_ALIGNMENT - 1) & ~(CONNX_ALIGNMENT - 1))

// Lifecycle
void connx_init();
void connx_destroy();

// Memory management
void* connx_alloc(uint32_t size);
void connx_free(void* ptr);

// Loaders
void* connx_load_model();
void connx_unload_model(void* buf);
void* connx_load_data(uint32_t graph_id, uint32_t id);
void connx_unload_data(void* buf);
void* connx_load_text(uint32_t graph_id);
void connx_unload_text(void* buf);

// Lock
#ifdef __linux__
typedef pthread_mutex_t connx_Lock;
#else
typedef struct connx_Lock {
    ;
} connx_Lock;
#endif

void connx_Lock_init(connx_Lock* lock);
void connx_Lock_destroy(connx_Lock* lock);

void connx_Lock_lock(connx_Lock* lock);
void connx_Lock_unlock(connx_Lock* lock);

// Thread pool
void connx_Thread_run_all(void* (*run)(void*), int32_t count, void* contexts, int32_t context_size);

// debugging message
struct _connx_Tensor;
struct _connx_Graph;
struct _connx_Node;
void connx_debug(const char* format, ...);
void connx_info(const char* format, ...);
void connx_error(const char* format, ...);

void connx_Iterator_dump(int32_t* iterator);
void connx_Tensor_dump(struct _connx_Tensor* tensor);
void connx_Tensor_dump_header(struct _connx_Tensor* tensor);

void connx_dump_node_outputs(struct _connx_Graph* graph, struct _connx_Node* node);

uint64_t connx_time();
void connx_watch_start(int32_t idx);
void connx_watch_stop(int32_t idx);
void connx_watch_dump();

#endif /* __CONNX_HAL_COMMON_H__ */
