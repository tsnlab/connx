.PHONY: all run test clean

CC := gcc
DEBUG ?= 1

override CFLAGS += -Iinclude -Wall -std=c99

ifeq ($(DEBUG), 1)
	override CFLAGS += -O0 -g -DDEBUG=1 -fsanitize=address
else
	override CFLAGS += -O3
endif

LIBS := -lm -pthread
#OPSET_OBJS := $(patsubst src/%.c, obj/%.o, $(wildcard src/opset_*.c))
OBJS := $(patsubst src/%.c, obj/%.o, $(wildcard src/*.c))

all: connx

run: all
	./connx examples/mnist

test: src/ver.h $(filter-out obj/main.o, $(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter %.o, $^) $(LIBS) -lcmocka
	./test

clean:
	rm -f src/ver.h
	rm -f src/opset.c
	rm -rf obj
	rm -f connx

connx: src/ver.h $(filter-out obj/test.o, $(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter %.o, $^) $(LIBS)

src/ver.h:
	bin/ver.sh

obj/%.d: src/ver.h src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -M $< > $@

-include $(patsubst src/%.c, obj/%.d, $(wildcard src/*.c))  

obj/%.o: src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -c -o $@ $^
