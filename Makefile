.PHONY: all

DEBUG ?= 1
OPSET ?= $(patsubst src/opset/%.c, %, $(wildcard src/opset/*))
HAL_SRC ?= ports/linux/src/hal.c
ACCEL_SRC ?= ports/linux/src/accel.c
OUT_DIR ?= gen

override CFLAGS += -Iinclude -std=c99

SRCS := $(wildcard src/*.c) $(patsubst %, src/opset/%.c, $(OPSET))
GENS:= $(patsubst src/%.c, $(OUT_DIR)/%.c, $(SRCS)) $(OUT_DIR)/opset.c $(OUT_DIR)/hal.c $(OUT_DIR)/accel.c

all: $(GENS)

$(OUT_DIR)/opset.c: | $(OUT_DIR)
	bin/opset.sh $(OPSET) > $@

$(OUT_DIR):
	mkdir -p $(OUT_DIR)/opset

$(OUT_DIR)/hal.c: $(HAL_SRC) | $(OUT_DIR)
	bin/preprocessor.py $< $@

$(OUT_DIR)/accel.c: $(ACCEL_SRC) | $(OUT_DIR)
	bin/preprocessor.py $< $@

$(OUT_DIR)/%.c: src/%.c | $(OUT_DIR)
	bin/preprocessor.py $< $@
