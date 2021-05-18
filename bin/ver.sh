#!/bin/bash

# Tag name must be v(major).(minor) format
TAG=`git describe --tags --long`

# Parse major, minor, micro and commit
MAJOR=`echo ${TAG:1} | awk -F- '{print $1}' | awk -F. '{print $1}'` 
MINOR=`echo ${TAG:1} | awk -F- '{print $1}' | awk -F. '{print $2}'` 
MICRO=`echo ${TAG:1} | awk -F- '{print $2}'`
COMMIT=`echo ${TAG:1} | awk -F- '{print $3}'`
COMMIT=${COMMIT:1:7}

cat << EOF > src/ver.h
#ifndef __CONNX_VERSION__
#define __CONNX_VERSION__

#define CONNX_VERSION_MAJOR ${MAJOR}
#define CONNX_VERSION_MINOR ${MINOR}
#define CONNX_VERSION_MICRO ${MICRO}
#define CONNX_VERSION_COMMIT ${COMMIT}
#define CONNX_VERSION "${MAJOR}.${MINOR}.${MICRO}-${COMMIT}"

#endif /* __CONNX_VERSION__ */
EOF
