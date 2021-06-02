#!/bin/bash

COUNT=${#@}

# Write header
cat << EOF
#include <connx/connx.h>

#define OPERATOR_COUNT ${COUNT}

EOF

# Write prototypes
for NAME in $@
do
cat << EOF
extern int ${NAME}(connx_Graph* graph, uint32_t* outputs, uint32_t* inputs, void** attributes);
EOF
done

# Write opset names
cat << EOF

char* connx_opset_names[] = {
EOF

for NAME in $@
do
cat << EOF
    "${NAME}",
EOF
done

cat << EOF
    NULL
};

EOF

# Write opset functions
cat << EOF

CONNX_OPERATOR connx_opset_ops[] = {
EOF

for NAME in $@
do
cat << EOF
    ${NAME},
EOF
done
cat << EOF
    NULL
};
EOF
