.PHONY: all run

CONNX_HOME := ../..
CWD := $(shell pwd)
OPSET ?= $(shell bin/opset_list.sh $(CONNX_HOME))
SRCS := src/hal.c src/accel.c src/main.c

all: gen


clean:
	# rm -rf gen

gen: src/accel.c
	../../bin/gen.sh -o gen -p src
	rm -rf gen/templates

