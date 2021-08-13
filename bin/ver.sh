#!/bin/bash

# Tag name must be v(major).(minor) format
if [[ -f ${HOME}/../TAG ]]; then
    TAG=`cat ${HOME}/../TAG`
else
    TAG=`git describe --tags --long`
fi

if [[ -z ${TAG} ]]; then
    TAG='v0.0-000-00000000'
fi

# Parse major, minor, micro and commit
MAJOR=`echo ${TAG:1} | awk -F- '{print $1}' | awk -F. '{print $1}'` 
MINOR=`echo ${TAG:1} | awk -F- '{print $1}' | awk -F. '{print $2}'` 
MICRO=`echo ${TAG:1} | awk -F- '{print $2}'`
COMMIT=`echo ${TAG:1} | awk -F- '{print $3}'`
COMMIT=${COMMIT:1:7}

cat << EOF
// ${TAG}
/*
 *  CONNX, C implementation of Open Neural Network Exchange Runtime
 *  Copyright (C) 2019-2021 TSN Lab, Inc.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef __CONNX_VERSION__
#define __CONNX_VERSION__

#define CONNX_VERSION_MAJOR ${MAJOR}
#define CONNX_VERSION_MINOR ${MINOR}
#define CONNX_VERSION_MICRO ${MICRO}
#define CONNX_VERSION_COMMIT "${COMMIT}"
#define CONNX_VERSION "${MAJOR}.${MINOR}.${MICRO}-${COMMIT}"

#endif /* __CONNX_VERSION__ */
EOF
