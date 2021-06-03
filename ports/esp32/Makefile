.PHONY: all run

CONNX_HOME := ../..
CWD := $(shell pwd)
OPSET ?= $(ls ../../src/opset/*.c | sed 's/\.\.\/\.\.\/src\/opset\///' | sed 's/\.c$//' | tr '\n' ' ')

all: build/connx-esp32.bin
	idf.py flash

run: 
	idf.py monitor

clean:
	rm -rf gen
	rm -rf spiffs
	rm -f main/CMakeFiles.txt
	rm -rf build

build/connx-esp32.bin: spiffs main/CMakeFiles.txt
	idf.py build

gen:
	make -C $(CONNX_HOME) OUT_DIR=$(CWD)/gen HAL_SRC=$(CWD)/main/hal.c ACCEL_SRC=$(CWD)/main/accel.c OPSET="$(OPSET)"

main/CMakeFiles.txt: gen
	bin/make_cmake.sh

spiffs:
	mkdir -p spiffs
	-cp $(CONNX_HOME)/test/data/node/test_asin/* spiffs 2> /dev/null
