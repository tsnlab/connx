.PHONY: all run clean cleanall mnist yolo3 mobilenet2

CC=gcc
CFLAGS=-I../include -I../lib/ds/include -Wall -g -O0 -fsanitize=address

LIBS=-L../lib/ds -lm -lds
OPSET_OBJS=$(patsubst ../src/%.c, obj/%.o, $(wildcard ../src/opset*.c)) obj/connx.o
OBJS=$(patsubst src/%.c, obj/%.o, $(wildcard src/*.c)) obj/parse.o

all: test

run: all
	./test

clean:
	rm -rf obj
	rm -f test

test: $(OBJS) $(OPSET_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

mnist:
	venv/bin/python rt_onnxruntime.py ../examples/mnist/model.onnx -i ../examples/mnist/test_data_set_0/input_0.pb -t ../examples/mnist/test_data_set_0/output_0.pb -d -e 0.01 -l 1000

yolo3:
	venv/bin/python rt_onnxruntime.py ../examples/yolov3-tiny/yolov3-tiny.onnx -i ../examples/yolov3-tiny/test_data_set_0/input_0.pb -i ../examples/yolov3-tiny/test_data_set_0/input_1.pb -t ../examples/yolov3-tiny/test_data_set_0/output_0.pb -t ../examples/yolov3-tiny/test_data_set_0/output_1.pb -t ../examples/yolov3-tiny/test_data_set_0/output_2.pb -d -e 0.001 -l 100

mobilenet2:
	venv/bin/python rt_onnxruntime.py ../examples/mobilenet2/mobilenetv2-7.onnx -i ../examples/mobilenet2/test_data_set_0/input_0.pb -t ../examples/mobilenet2/test_data_set_0/output_0.pb -d -e 0.001 -l 10

src/parse.c: src/parse.re
	../bin/re2c -T $^ -o $@

obj/%.d : src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -M $< > $@

obj/%.d : ../src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -M $< > $@

-include $(patsubst obj/%.o, obj/%.d, $(OBJS))  
-include $(patsubst obj/%.o, obj/%.d, $(OPSET_OBJS))  

obj/parse.o: src/parse.c
	mkdir -p obj; $(CC) $(CFLAGS) -Wno-unused-variable -c -o $@ $^

obj/%.o: src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -c -o $@ $^

obj/%.o: ../src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -c -o $@ $^
