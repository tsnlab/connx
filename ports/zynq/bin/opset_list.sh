#!/bin/bash

LIST=`ls $1/src/opset/*.c`
OPSET=''

for NAME in $LIST; do
	NAME=${NAME#"$1/src/opset/"}
	NAME=${NAME%".c"}
	OPSET="$OPSET $NAME"
done

echo $OPSET
