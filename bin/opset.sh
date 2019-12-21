#!/bin/bash

echo "opset: $@"

LIST=()

for OBJ in $@
do
	LIST+=(`echo $OBJ | sed -e 's/obj\/opset_//' | sed -e 's/\.o$//'`)
done

COUNT=${#LIST[@]}

cat << EOF > src/opset.c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <connx/connx.h>

#define OPERATOR_COUNT	${COUNT}

uint32_t connx_operator_count;
connx_Operator connx_operators[OPERATOR_COUNT];

bool connx_init() {
EOF

for (( i = 0; i < $COUNT; i++))
do
cat << EOF >> src/opset.c
	bool connx_opset_${LIST[$i]}_init();
	if(!connx_opset_${LIST[$i]}_init()) {
		return false;
	}
EOF
done

cat << EOF >> src/opset.c
	return true;
}
EOF
