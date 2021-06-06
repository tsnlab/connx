.PHONY: all run test perf clean

CONNX_HOME ?= ../..
CC := gcc
AR := ar
DEBUG ?= 1
OPSET ?= $(patsubst $(CONNX_HOME)/src/opset/%.c, %, $(wildcard $(CONNX_HOME)/src/opset/*))
DUMMY := $(shell make -C $(CONNX_HOME) OUT_DIR=$(shell pwd)/gen HAL_SRC=$(shell pwd)/src/hal.c ACCEL_SRC=$(shell pwd)/src/accel.c OPSET='$(OPSET)')
SRCS := $(wildcard gen/*.c) $(wildcard gen/opset/*.c)
OBJS := $(patsubst gen/%.c, obj/%.o, $(SRCS))

override CFLAGS += -I../../include -Wall -std=c99

ifeq ($(DEBUG), 1)
	override CFLAGS += -pg -O0 -g -DDEBUG=1 -fsanitize=address
else
	override CFLAGS += -O3
endif

LIBS := -lm -pthread

all: connx

run: connx
	# Run Asin test case
	rm -f tensorin
	rm -f tensorout
	mkfifo tensorin
	mkfifo tensorout
	# Run connx as daemon
	./connx  $(CONNX_HOME)/examples/mnist/ tensorin tensorout &
	# Run the test
	python3 $(CONNX_HOME)/bin/run.py tensorin tensorout $(CONNX_HOME)/examples/mnist/test_data_set_0/input-0_1_4_1_1_28_28.data
	# Shutdown the daemon
	python3 $(CONNX_HOME)/bin/run.py tensorin tensorout
	# Clean
	rm -f tensorin
	rm -f tensorout

connx: $(OBJS) obj/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

test: connx
	python3 $(CONNX_HOME)/bin/test.py ./connx $(CONNX_HOME)

perf:
	gprof ./connx gmon.out

clean:
	rm -rf obj
	rm -rf gen
	rm -f connx
	rm -f gmon.out

obj:
	mkdir -p obj/opset

obj/%.o: gen/%.c | obj
	$(CC) $(CFLAGS) -c -o $@ $<

obj/main.o: src/main.c | obj
	$(CC) $(CFLAGS) -c -o $@ $<

ifneq (clean, $(filter clean, $(MAKECMDGOALS)))
-include $(DEPS)
endif
