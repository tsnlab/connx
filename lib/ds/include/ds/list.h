#ifndef __DS_LIST__
#define __DS_LIST__

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

struct _ds_ListNode {
	struct _ds_ListNode*	prev;
	struct _ds_ListNode*	next;
	uint8_t					base[0];
};

struct ds_List {
	void*(*alloc)(size_t);
	void(*free)(void*);

	bool(*equals)(void*, void*);

	uint32_t				size;
	uint32_t				count;

	struct _ds_ListNode*	head;
	struct _ds_ListNode*	tail;
};

struct ds_ListIterator {
	struct ds_List*			list;
	struct _ds_ListNode*	node;
	struct _ds_ListNode*	prev;
};

bool ds_List_equals_string(void* v1, void* v2);
bool ds_List_equals_ptr(void* v1, void* v2);

struct ds_List* ds_List_create(uint32_t size, bool(*equals)(void*, void*), void*(*alloc)(size_t), void(*free)(void*));
void ds_List_delete(struct ds_List* list);
bool ds_List_add(struct ds_List* list, void* ptr);
void* ds_List_get(struct ds_List* list, uint32_t idx);
bool ds_List_remove(struct ds_List* list, void* ptr);
bool ds_List_removeAt(struct ds_List* list, uint32_t idx, void* ptr);
bool ds_List_contains(struct ds_List* list, void* ptr);
uint32_t ds_List_clear(struct ds_List* list);

struct ds_ListIterator ds_ListIterator_create(struct ds_List* list);
bool ds_ListIterator_hasNext(struct ds_ListIterator* iter);
void* ds_ListIterator_next(struct ds_ListIterator* iter);
bool ds_ListIterator_remove(struct ds_ListIterator* iter);

#endif /* __DS_LIST__ */
