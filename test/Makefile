.PHONY: all run clean cleanall

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

src/parse.c: src/parse.re
	../bin/re2c -T $^ -o $@

obj/%.d : src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -M $< > $@

-include $(patsubst obj/%.o, obj/%.d, $(OBJS))  

obj/parse.o: src/parse.c
	mkdir -p obj; $(CC) $(CFLAGS) -Wno-unused-variable -c -o $@ $^

obj/%.o: src/%.c
	mkdir -p obj; $(CC) $(CFLAGS) -c -o $@ $^
