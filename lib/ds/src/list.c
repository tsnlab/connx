#include <string.h>
#include <ds/list.h>

bool ds_List_equals_string(void* v1, void* v2) {
	return strcmp(v1, v2) == 0;
}

bool ds_List_equals_ptr(void* v1, void* v2) {
	return v1 == v2;
}

struct ds_List* ds_List_create(uint32_t size, bool(*equals)(void*, void*), void*(*alloc)(size_t), void(*free)(void*)) {
	struct ds_List* list = alloc(sizeof(struct ds_List));
	if(list == NULL)
		return NULL;

	list->alloc = alloc;
	list->free = free;

	if(equals == NULL)
		list->equals = ds_List_equals_ptr;
	else
		list->equals = equals;

	list->head = NULL;
	list->tail = NULL;
	list->size = size;
	list->count = 0;

	return list;
}

void ds_List_delete(struct ds_List* list) {
	struct _ds_ListNode* node = list->head;
	while(node != NULL) {
		struct _ds_ListNode* next = node->next;
		list->free(node);

		node = next;
	}

	list->free(list);
}

bool ds_List_add(struct ds_List* list, void* ptr) {
	struct _ds_ListNode* node = list->alloc(sizeof(struct _ds_ListNode) + list->size);
	if(node == NULL)
		return false;

	*(uintptr_t*)node->base = (uintptr_t)ptr;

	if(list->tail == NULL) {
		list->head = list->tail = node;
	} else {
		list->tail->next = node;
		node->prev = list->tail;
		list->tail = node;
	}

	list->count++;

	return true;
}

void* ds_List_get(struct ds_List* list, uint32_t idx) {
	uint32_t i = 0;
	struct _ds_ListNode* node = list->head;
	while(node != NULL) {
		if(i++ == idx) {
			return (void*)*(uintptr_t*)node->base;
		}

		node = node->next;
	}

	return NULL;
}

static void _ds_List_remove(struct ds_List* list, struct _ds_ListNode* node) {
	if(node->prev != NULL)
		node->prev->next = node->next;

	if(node->next != NULL)
		node->next->prev = node->prev;

	if(list->head == node)
		list->head = node->next;

	if(list->tail == node)
		list->tail = node->prev;

	list->free(node);

	list->count--;
}

bool ds_List_remove(struct ds_List* list, void* ptr) {
	struct _ds_ListNode* node = list->head;
	while(node != NULL) {
		if(list->equals((void*)*(uintptr_t*)node->base, ptr)) {
			_ds_List_remove(list, node);

			return true;
		}

		node = node->next;
	}

	return false;
}

bool ds_List_removeAt(struct ds_List* list, uint32_t idx, void* ptr) {
	struct _ds_ListNode* node = list->head;
	while(idx-- != 0) {
		if(node == NULL)
			return false;

		node = node->next;
	}

	if(ptr != NULL)
		*(uintptr_t*)ptr = *(uintptr_t*)node->base;

	_ds_List_remove(list, node);

	return true;
}

bool ds_List_contains(struct ds_List* list, void* ptr) {
	struct _ds_ListNode* node = list->head;
	while(node != NULL) {
		if(list->equals((void*)*(uintptr_t*)node->base, ptr))
			return true;

		node = node->next;
	}

	return false;
}

uint32_t ds_List_clear(struct ds_List* list) {
	uint32_t count = list->count;

	struct _ds_ListNode* node = list->head;
	while(node != NULL) {
		struct _ds_ListNode* next = node->next;
		list->free(node);
		node = next;
	}

	list->head = list->tail = NULL;
	list->count = 0;

	return count;
}

struct ds_ListIterator ds_ListIterator_create(struct ds_List* list) {
	struct ds_ListIterator iter;
	iter.list = list;
	iter.node = list->head;
	iter.prev = NULL;

	return iter;
}

bool ds_ListIterator_hasNext(struct ds_ListIterator* iter) {
	return iter->node != NULL;
}

void* ds_ListIterator_next(struct ds_ListIterator* iter) {
	void* ptr = (void*)*(uintptr_t*)iter->node->base;
	iter->prev = iter->node;
	iter->node = iter->node->next;

	return ptr;
}

bool ds_ListIterator_remove(struct ds_ListIterator* iter) {
	if(iter->prev != NULL) {
		struct _ds_ListNode* prev = iter->prev;
		iter->prev = prev->prev;

		_ds_List_remove(iter->list, prev);
		return true;
	} else {
		return false;
	}
}

