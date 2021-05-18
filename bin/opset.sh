#!/bin/bash

echo "opset: $@"

COUNT=${#@}

# Write header
cat << EOF > src/opset.c
#include "connx.h"

#define OPERATOR_COUNT ${COUNT}

EOF

# Write prototypes
for NAME in $@
do
cat << EOF >> src/opset.c
extern int ${NAME}(connx_Graph* graph, uint32_t* outputs, uint32_t* inputs, void** attributes);
EOF
done

# Write opset names
cat << EOF >> src/opset.c

char* connx_opset_names[] = {
EOF

for NAME in $@
do
cat << EOF >> src/opset.c
    "${NAME}",
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

for NAME in $@
do
cat << EOF >> src/opset.c
	${NAME},
EOF
done
cat << EOF >> src/opset.c
	NULL
};
EOF
