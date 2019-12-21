.PHONY: all clean cleanall run

CC=gcc
CFLAGS=-Iinclude -Wall -g -O0 -std=c99 -fsanitize=address
#CFLAGS=-Iinclude -Wall -O3 -std=c99
LIBS=-lprotobuf-c
OBJS=$(patsubst src/%.c, obj/%.o, $(wildcard src/*.c)) obj/onnx.proto3.pb-c.o

all: main

run: all
	./main

clean:
	rm -rf obj
	rm -f main

main: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clear:
	rm src/onnx.proto3.pb-c.c
	rm include/onnx/onnx.proto3.pb-c.h
	rmdir include/onnx
	rm main

src/onnx.proto3.pb-c.c: onnx/onnx/onnx.proto3
	cd onnx; protoc-c onnx/onnx.proto3 --c_out ../src
	mv src/onnx/onnx.proto3.pb-c.c src/
	mv src/onnx include/

obj/%.d : src/%.c  
	mkdir -p obj; $(CC) $(CFLAGS) -M $< > $@

-include $(patsubst src/%.c, obj/%.d, $(wildcard src/*.c))  

obj/%.o: src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -c -o $@ $^
