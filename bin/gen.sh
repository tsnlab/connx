#!/bin/bash

HOME=`dirname $0`
INPUT_DIR=../../src
OUTPUT_DIR=build/src
PORT_DIR=src
OPERATORS=-all
IS_DUMP=0

if [ -f 'opset.txt' ]; then
    OPERATORS=`cat opset.txt`
fi

while (( "$#" )); do
    case "$1" in
        -i|--input)
            INPUT_DIR=$2
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR=$2
            shift 2
            ;;
        -p|--port)
            PORT_DIR=$2
            shift 2
            ;;
        -s|--set)
            if [ -f $2 ]; then
                OPERATORS=`cat $2`
            fi
            shift 2
            ;;
        -d|--dump)
            IS_DUMP=1
            shift 1
            ;;
        -h|--help)
            echo "Usage: gen.sh -i [connx source directory] -o [generated code directory] -p [port directory] -s [opset file] [[-d|--dump]]"
            echo "    [connx source directory] - ../../src"
            echo "    [generated code directory] - build/src"
            echo "    [port directory] - src"
            echo "    [opset file] - opset.txt"
            exit 0
            ;;
        *)
            OPERATORS=$*
            shift $#
            ;;
    esac
done

# parse opset
OPSET=

for NAME in ${OPERATORS}; do
    if [ ${NAME} == "-all" ]; then
        FILES=`ls ${INPUT_DIR}/opset/*.c`
        for FILE in ${FILES}; do
            FILE=${FILE#${INPUT_DIR}/opset/}
            FILE=${FILE%.c}
            OPSET="${OPSET} ${FILE}"
        done
    else
        FILE=`ls ${INPUT_DIR}/opset/${NAME}.c 2> /dev/null`
        FILE=${FILE#${INPUT_DIR}/opset/}
        FILE=${FILE%.c}
        OPSET="${OPSET} ${FILE}"
    fi
done

CONNX=

for FILE in `ls ${INPUT_DIR}/*.c`; do
    FILE=${FILE#${INPUT_DIR}/}
    CONNX="${CONNX} ${FILE}"
done

#echo "HOME=${HOME}"
#echo "INPUT_DIR=${INPUT_DIR}"
#echo "OUTPUT_DIR=${OUTPUT_DIR}"
#echo "PORT_DIR=${PORT_DIR}"
#echo "OPSET=${OPSET}"
#echo "CONNX=${CONNX}"

if [[ ${IS_DUMP} == 0 ]]; then
    mkdir -p ${OUTPUT_DIR}/opset
fi

# Copy connx codes
for FILE in ${CONNX}; do
    if [[ ${IS_DUMP} == 1 ]]; then
        echo "${FILE}"
    else
        if [[ ! -f ${OUTPUT_DIR}/${FILE} ]] || [[ ${INPUT_DIR}/${FILE} -nt ${OUTPUT_DIR}/${FILE} ]]; then
            echo "Copying ${OUTPUT_DIR}/${FILE}"
            cp ${INPUT_DIR}/${FILE} ${OUTPUT_DIR}/${FILE}
        fi
    fi
done

# Generate opset.c
if [[ ${IS_DUMP} == 1 ]]; then
    echo "opset.c"
else
    if [[ ! -f ${OUTPUT_DIR}/opset.c ]] || [[ "//${OPSET}" != `head -1 ${OUTPUT_DIR}/opset.c` ]]; then
        echo "Generating ${OUTPUT_DIR}/opset.c"
        $HOME/opset.sh ${OPSET} > ${OUTPUT_DIR}/opset.c
    fi
fi

# Generate ver.h
if [[ -f ${HOME}/../TAG ]]; then
    TAG=`cat ${HOME}/../TAG`
else:
    TAG=`git describe --tags --long`
fi

if [[ -z ${TAG} ]]; then
    TAG='v0.0-000-00000000'
fi

if [[ ${IS_DUMP} == 1 ]]; then
    echo "ver.h"
else
    if [[ ! -f ${OUTPUT_DIR}/ver.h ]] || [[ "// ${TAG}" != `head -1 ${OUTPUT_DIR}/ver.h` ]]; then
        echo "Generating ${OUTPUT_DIR}/ver.h with tag ${TAG}"
        $HOME/ver.sh > ${OUTPUT_DIR}/ver.h
    fi
fi

# Generate opset codes
for FILE in ${OPSET}; do
    if [[ ${IS_DUMP} == 1 ]]; then
        echo "opset/${FILE}.c"
    else
        if [[ ${INPUT_DIR}/opset/${FILE}.c -nt ${OUTPUT_DIR}/opset/${FILE}.c ]]; then
            echo "Preprocessing ${OUTPUT_DIR}/opset/${FILE}.c"
            ${HOME}/preprocessor.py ${INPUT_DIR}/opset/${FILE}.c ${OUTPUT_DIR}/opset/${FILE}.c
        fi
    fi
done

# Generate port codes
for FILE in `ls ${PORT_DIR}/*.c`; do
    FILE=${FILE#${PORT_DIR}/}

    if [[ ${IS_DUMP} == 1 ]]; then
        echo "${FILE}"
    else
        if [[ ${PORT_DIR}/${FILE} -nt ${OUTPUT_DIR}/${FILE} ]]; then
            echo "Preprocessing ${OUTPUT_DIR}/${FILE}"
            ${HOME}/preprocessor.py ${PORT_DIR}/${FILE} ${OUTPUT_DIR}/${FILE}
        fi
    fi
done
