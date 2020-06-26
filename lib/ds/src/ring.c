#include <ds/ring.h>

struct ds_Ring* ds_Ring_create(uint32_t size, uint32_t count, void*(*alloc)(size_t), void(*free)(void*)) {
	struct ds_Ring* ring = alloc(sizeof(struct ds_Ring) * (size * count));
	if(ring == NULL)
		return NULL;

	ring->alloc = alloc;
	ring->free = free;

	ring->head = 0;
	ring->tail = 0;
	ring->size = size;
	ring->count = count;

	return ring;
}

void ds_Ring_delete(struct ds_Ring* ring) {
	ring->free(ring);
}

bool ds_Ring_enqueue(struct ds_Ring* ring, void* ptr) {
	uint32_t next = (ring->tail + 1) % ring->count;
	if(next == ring->head)
		return false;

	((uintptr_t*)ring->base)[ring->tail] = (uintptr_t)ptr;
	ring->tail = next;

	return true;
}

bool ds_Ring_dequeue(struct ds_Ring* ring, void* ptr) {
	if(ring->head == ring->tail)
		return false;

	*(uintptr_t*)ptr = ((uintptr_t*)ring->base)[ring->head];
	ring->head = (ring->head + 1) % ring->count;

	return true;
}

