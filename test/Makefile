.PHONY: all run clean cleanall example_yolo

CC=gcc
CFLAGS=-I../include -Wall -g -O0 -fsanitize=address

LIBS=-lm
OPSET_OBJS=$(patsubst ../src/%.c, ../obj/%.o, $(wildcard ../src/opset*.c)) ../obj/connx.o
OBJS=$(patsubst src/%.c, obj/%.o, $(wildcard src/*.c)) obj/parse.o

all: test

run: all
	./test

clean:
	rm -rf obj
	rm -f test

test: $(OBJS) $(OPSET_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

example_yolo:
	python rt_onnxruntime.py ../examples/yolov3-tiny/yolov3-tiny.onnx -i ../examples/yolov3-tiny/test_data_set_0/input_0.pb -i ../examples/yolov3-tiny/test_data_set_0/input_1.pb -t ../examples/yolov3-tiny/test_data_set_0/output_0.pb -t ../examples/yolov3-tiny/test_data_set_0/output_1.pb -t ../examples/yolov3-tiny/test_data_set_0/output_2.pb -d -l 1 -e 0.001

src/parse.c: src/parse.re
	../bin/re2c -T $^ -o $@

obj/%.d : src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -M $< > $@

-include $(patsubst obj/%.o, obj/%.d, $(OBJS))  

obj/parse.o: src/parse.c
	mkdir -p obj; $(CC) $(CFLAGS) -Wno-unused-variable -c -o $@ $^

obj/%.o: src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -c -o $@ $^
