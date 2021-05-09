.PHONY: all run test clean

CC := gcc
DEBUG ?= 1
OPSET ?= $(patsubst src/opset/%.c, %, $(wildcard src/opset/*))

override CFLAGS += -Iinclude -Wall -std=c99

ifeq ($(DEBUG), 1)
	override CFLAGS += -O0 -g -DDEBUG=1 -fsanitize=address
else
	override CFLAGS += -O3
endif

LIBS := -lm -pthread
SRCS := $(wildcard src/*.c) src/opset.c $(patsubst %, src/opset/%.c, $(OPSET))
OBJS := $(patsubst src/%.c, obj/%.o, $(SRCS))
DEPS := $(patsubst src/%.c, obj/%.d, $(SRCS))

all: connx

run: all
	./connx examples/mnist

test: src/ver.h $(filter-out obj/main.o, $(OBJS))
	#$(CC) $(CFLAGS) -o $@ $(filter %.o, $^) $(LIBS) -lcmocka
	python bin/run.py ./connx testcase/data/node/test_asin/ testcase/data/node/test_asin/test_data_set_0/input-0_1_3_3_4_5.data

clean:
	rm -f src/ver.h
	rm -f src/opset.c
	rm -rf obj
	rm -f connx test

connx: src/ver.h $(filter-out obj/test.o, $(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter %.o, $^) $(LIBS)

src/ver.h:
	bin/ver.sh

src/opset.c:
	bin/opset.sh $(OPSET)

obj/%.d: src/ver.h $(SRCS)
	mkdir -p obj/opset; $(CC) $(CFLAGS) -M $< > $@

obj/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

-include $(DEPS)
