#!/bin/bash

echo "opset: $@"

LIST=()

for NAME in $@
do
	LIST+=$NAME
done

COUNT=${#LIST[@]}

# Write header
cat << EOF > src/opset.c
#include "connx.h"

#define OPERATOR_COUNT	${COUNT}

EOF

# Write prototypes
for (( i = 0; i < $COUNT; i++))
do
cat << EOF >> src/opset.c
extern int ${LIST[$i]}(connx_Graph* graph, uint32_t* outputs, uint32_t* inputs, void** attributes);
EOF
done

# Write opset names
cat << EOF >> src/opset.c

char* connx_opset_names[] = {
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

# Write opset functions
cat << EOF >> src/opset.c

CONNX_OPERATOR connx_opset_ops[] = {
EOF

for (( i = 0; i < $COUNT; i++))
do
cat << EOF >> src/opset.c
	${LIST[$i]},
EOF
done
cat << EOF >> src/opset.c
	NULL
};
EOF
