.PHONY: all test clean

RELEASE ?= 1
CC := gcc
AR := ar
override CFLAGS += -Iinclude -Wall -std=c99
ifeq ($(RELEASE), 1)
	override CFLAGS += -O3
else
	override CFLAGS += -O0 -g
endif

OBJS := $(patsubst src/%.c, obj/%.o, $(wildcard src/*.c))

all: libds.a

clean:
	rm -rf obj
	rm -f libds.a

libds.a: $(OBJS)
	$(AR) rcs $@ $^

-include $(patsubst src/%.c, obj/%.d, $(wildcard src/*.c))  

obj/%.o: src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -c -o $@ $^
