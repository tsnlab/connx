#!/bin/bash

echo "opset: $@"

LIST=()

for OBJ in $@
do
	LIST+=(`echo $OBJ | sed -e 's/obj\/opset_//' | sed -e 's/\.o$//'`)
done

COUNT=${#LIST[@]}

cat << EOF > src/opset.c
#include <connx/operator.h>

#define OPERATOR_COUNT	${COUNT}

char* connx_operator_names[] = {
EOF
for (( i = 0; i < $COUNT; i++))
do
cat << EOF >> src/opset.c
	"${LIST[$i]}",
EOF
done

cat << EOF >> src/opset.c
	NULL
};

EOF
for (( i = 0; i < $COUNT; i++))
do
cat << EOF >> src/opset.c
extern bool opset_${LIST[$i]}(connx_Backend* backend, uint32_t counts, uint32_t* params);
EOF
done

cat << EOF >> src/opset.c

connx_Operator connx_operators[] = {
EOF
for (( i = 0; i < $COUNT; i++))
do
cat << EOF >> src/opset.c
	opset_${LIST[$i]},
EOF
done
cat << EOF >> src/opset.c
	NULL
};
EOF
