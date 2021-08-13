.PHONY: all run

CONNX_HOME := ../..
CWD := $(shell pwd)
OPSET ?= $(shell bin/opset_list.sh $(CONNX_HOME))
SRCS := main/hal.c main/accel.c main/main.c main/cam.c

all: build/connx-esp32.bin
	idf.py flash

run: build/connx-esp32.bin
	idf.py monitor

clean:
	rm -rf gen
	rm -rf spiffs
	rm -f main/CMakeFiles.txt
	rm -rf build

build/connx-esp32.bin: spiffs main/CMakeFiles.txt $(SRCS)
	idf.py build

gen: main/hal.c main/accel.c
	../../bin/gen.sh -o gen -p main

main/CMakeFiles.txt: gen
	bin/make_cmake.sh

spiffs:
	mkdir -p spiffs
	-cp $(CONNX_HOME)/examples/mnist/* -r spiffs 2> /dev/null
