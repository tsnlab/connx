.PHONY: all clean cleanall run

RELEASE ?= 0
CC=gcc
ifeq ($(RELEASE), 1)
	CFLAGS=-Iinclude -Wall -O3 -std=c99
else
	CFLAGS=-Iinclude -Wall -g -O0 -fsanitize=address
endif

LIBS=-lprotobuf-c
OPSET_OBJS=$(patsubst src/%.c, obj/%.o, $(wildcard src/opset_*.c))
OBJS=$(patsubst src/%.c, obj/%.o, $(wildcard src/*.c)) obj/opset.o obj/onnx.proto3.pb-c.o

all: connx

run: all
	./connx

clean:
	rm src/opset.c
	rm -rf obj
	rm -f connx

cleanall: clean
	rm src/onnx.proto3.pb-c.c
	rm include/onnx/onnx.proto3.pb-c.h
	rmdir include/onnx

connx: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)


src/opset.c:
	bin/opset.sh $(OPSET_OBJS)

src/onnx.proto3.pb-c.c: onnx/onnx/onnx.proto3
	cd onnx; protoc-c onnx/onnx.proto3 --c_out ../src
	mv src/onnx/onnx.proto3.pb-c.c src/
	mv src/onnx include/

obj/%.d : src/%.c src/onnx.proto3.pb-c.c
	mkdir -p obj; $(CC) $(CFLAGS) -M $< > $@

-include $(patsubst src/%.c, obj/%.d, $(wildcard src/*.c))  

obj/%.o: src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -c -o $@ $^
