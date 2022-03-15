.PHONY: all run fclean clean build

OPSET ?= $(shell bin/opset_list.sh)

all: gen

clean:
	rm -rf gen .metadata .Xil design_1_wrapper gen test test_system .analytics IDE.log *.str *.log RemoteSystemsTempFiles

gen: src/accel.c
	../../bin/gen.sh -o gen -p src
	rm -rf gen/templates

build: gen
	$(shell bin/build.sh > /dev/null)

run:
	$(shell bin/run.sh > /dev/null)
