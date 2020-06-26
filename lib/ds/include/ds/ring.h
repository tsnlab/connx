#ifndef __DS_RING__
#define __DS_RING__

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

struct ds_Ring {
	void*(*alloc)(size_t);
	void(*free)(void*);

	uint32_t	head;
	uint32_t	tail;
	uint32_t	size;
	uint32_t	count;
	uint8_t		base[0];
};

struct ds_Ring* ds_Ring_create(uint32_t size, uint32_t count, void*(*alloc)(size_t), void(*free)(void*));
void ds_Ring_delete(struct ds_Ring* ring);

bool ds_Ring_enqueue(struct ds_Ring* ring, void* ptr);
bool ds_Ring_dequeue(struct ds_Ring* ring, void* ptr);

#endif /* __DS_RING__ */
