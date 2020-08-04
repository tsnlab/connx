.PHONY: all clean run mnist mobilenet

RELEASE ?= 0
CC := gcc
override CFLAGS += -Iinclude -Wall -std=c99
ifeq ($(RELEASE), 1)
	override CFLAGS += -O3
else
	override CFLAGS += -O0 -g -fsanitize=address
endif

LIBS := -lm -pthread
OPSET_OBJS := $(patsubst src/%.c, obj/%.o, $(wildcard src/opset_*.c))
OBJS := $(patsubst src/%.c, obj/%.o, $(wildcard src/*.c)) obj/opset.o

all: connx

run: mnist

mnist: all
	./connx examples/mnist -i input_0.tensor -t output_0.tensor -l 1000

mobilenet: all
	./connx examples/mobilenet -i input_0.tensor -t output_0.tensor -l 1000

clean:
	rm -f src/ver.h
	rm -f src/opset.c
	rm -rf obj
	rm -f connx

connx: src/ver.h $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(filter %.o, $^) $(LIBS)

src/ver.h:
	bin/ver.sh

src/opset.c: bin/opset.sh $(OPSET_OBJS)
	bin/opset.sh $(OPSET_OBJS)

obj/%.d: src/ver.h src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -M $< > $@

-include $(patsubst src/%.c, obj/%.d, $(wildcard src/*.c))  

obj/connx.o: src/connx.c
	mkdir -p obj; $(CC) $(CFLAGS) -c -o $@ $^

obj/%.o: src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -c -o $@ $^
